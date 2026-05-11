#ifndef Nott_REPAIR_HPP
#define Nott_REPAIR_HPP
/// 1) HoloClean: Holistic Data Repairs with Probabilistic Inference https://arxiv.org/pdf/1702.00820
#include <algorithm>
#include <cmath>
#include <functional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Nott::Data::Repair {
    class HoloClean {
    public:
        struct Record {
            std::string id;
            std::unordered_map<std::string, std::string> values;
        };

        struct FunctionalDependency {
            std::vector<std::string> determinants;
            std::string dependent;
            double weight{1.0};
        };

        struct CellCoordinate {
            std::size_t record_index{};
            std::string attribute;

            bool operator==(const CellCoordinate &other) const {
                return record_index == other.record_index && attribute == other.attribute;
            }
        };

        struct CellCoordinateHasher {
            std::size_t operator()(const CellCoordinate &cell) const {
                std::size_t h1 = std::hash<std::size_t>{}(cell.record_index);
                std::size_t h2 = std::hash<std::string>{}(cell.attribute);
                return h1 ^ (h2 << 1);
            }
        };

        struct Candidate {
            CellCoordinate cell;
            std::string value;
            std::unordered_map<std::string, double> feature_scores;
            double probability{0.0};
        };

        struct RepairResult {
            CellCoordinate cell;
            std::string original_value;
            std::string repaired_value;
            double confidence{0.0};
            bool applied{false};
        };

        HoloClean() = default;

        /// Configure the probability threshold that a candidate must exceed to applied as a repair.
        void set_repair_threshold(double threshold) { repair_threshold_ = threshold; }

        /**
         * Register a functional dependency constraint.  Determinants correspond to
         * the left-hand-side attributes and dependent is the right-hand-side
         * attribute.  The weight is used as a multiplier for the resulting feature
         * during inference.
         */
        void add_functional_dependency(FunctionalDependency fd) {
            functional_dependencies_.push_back(std::move(fd));
        }

        std::size_t add_record(Record record) {
            records_.push_back(std::move(record));
            return records_.size() - 1;
        }

        void run() {
            dirty_cells_.clear();
            candidates_.clear();
            results_.clear();

            detect_errors();
            generate_candidates();
            build_factor_graph();
            run_inference();
            apply_repairs();
        }

        const std::vector<RepairResult> &results() const { return results_; }

    private:
        using GroupKey = std::string;

        void detect_errors() {
            for (const auto &fd : functional_dependencies_) {
                std::unordered_map<GroupKey, std::set<std::string>> group_to_values;
                std::unordered_map<GroupKey, std::vector<std::size_t>> group_to_indices;

                for (std::size_t idx = 0; idx < records_.size(); ++idx) {
                    const auto &record = records_[idx];
                    GroupKey key = build_group_key(record, fd.determinants);
                    const auto dependent_iter = record.values.find(fd.dependent);
                    if (dependent_iter == record.values.end()) {
                        continue;
                    }

                    group_to_values[key].insert(dependent_iter->second);
                    group_to_indices[key].push_back(idx);
                }

                for (const auto &[key, values] : group_to_values) {
                    if (values.size() > 1) {
                        for (auto idx : group_to_indices[key]) {
                            dirty_cells_.insert(CellCoordinate{idx, fd.dependent});
                        }
                    }
                }
            }
        }

        void generate_candidates() {
            std::unordered_map<std::string, std::unordered_map<std::string, std::size_t>> attribute_value_counts;
            for (const auto &record : records_) {
                for (const auto &[attribute, value] : record.values) {
                    attribute_value_counts[attribute][value]++;
                }
            }

            for (const auto &cell : dirty_cells_) {
                const auto &record = records_[cell.record_index];
                const auto original_iter = record.values.find(cell.attribute);
                if (original_iter == record.values.end()) {
                    continue;
                }

                std::set<std::string> observed_candidates;
                for (const auto &fd : functional_dependencies_) {
                    if (fd.dependent != cell.attribute) {
                        continue;
                    }

                    GroupKey key = build_group_key(record, fd.determinants);
                    collect_candidates_from_group(cell, key, fd, observed_candidates);
                }

                for (const auto &value_count : attribute_value_counts[cell.attribute]) {
                    if (value_count.first != original_iter->second) {
                        observed_candidates.insert(value_count.first);
                    }
                }

                for (const auto &value : observed_candidates) {
                    Candidate candidate;
                    candidate.cell = cell;
                    candidate.value = value;
                    candidates_.push_back(std::move(candidate));
                }
            }
        }

        void collect_candidates_from_group(const CellCoordinate &cell,
                                           const GroupKey &group_key,
                                           const FunctionalDependency &fd,
                                           std::set<std::string> &values) const {
            std::unordered_map<GroupKey, std::set<std::string>> group_values;

            for (const auto &record : records_) {
                GroupKey key = build_group_key(record, fd.determinants);
                const auto dependent_iter = record.values.find(fd.dependent);
                if (dependent_iter == record.values.end()) {
                    continue;
                }

                group_values[key].insert(dependent_iter->second);
            }

            auto it = group_values.find(group_key);
            if (it == group_values.end()) {
                return;
            }

            values.insert(it->second.begin(), it->second.end());
            const auto &record = records_[cell.record_index];
            auto original_iter = record.values.find(cell.attribute);
            if (original_iter != record.values.end()) {
                values.erase(original_iter->second);
            }
        }

        void build_factor_graph() {
            std::unordered_map<std::string, std::unordered_map<std::string, std::size_t>> attribute_value_counts;
            for (const auto &record : records_) {
                for (const auto &[attribute, value] : record.values) {
                    attribute_value_counts[attribute][value]++;
                }
            }

            for (auto &candidate : candidates_) {
                const auto &record = records_[candidate.cell.record_index];
                const auto original_value = record.values.at(candidate.cell.attribute);

                double prior_frequency = 0.0;
                auto attribute_iter = attribute_value_counts.find(candidate.cell.attribute);
                if (attribute_iter != attribute_value_counts.end()) {
                    const auto &value_counts = attribute_iter->second;
                    auto count_iter = value_counts.find(candidate.value);
                    if (count_iter != value_counts.end()) {
                        prior_frequency = static_cast<double>(count_iter->second) /
                                          static_cast<double>(records_.size());
                    }
                }
                candidate.feature_scores["prior_frequency"] = prior_frequency;

                for (const auto &fd : functional_dependencies_) {
                    if (fd.dependent != candidate.cell.attribute) {
                        continue;
                    }

                    GroupKey key = build_group_key(record, fd.determinants);
                    double support = functional_dependency_support(fd, key, candidate.value);
                    candidate.feature_scores["fd:" + fd.dependent] = support * fd.weight;
                }

                candidate.feature_scores["edit_distance"] =
                    1.0 - normalized_levenshtein(original_value, candidate.value);
            }
        }

        void run_inference() {
            for (auto &candidate : candidates_) {
                double linear_combination = 0.0;
                for (const auto &[feature, score] : candidate.feature_scores) {
                    double weight = feature_weights(feature);
                    linear_combination += weight * score;
                }
                candidate.probability = logistic(linear_combination);
            }
        }

        void apply_repairs() {
            std::unordered_map<CellCoordinate, Candidate, CellCoordinateHasher> best_candidate;
            for (const auto &candidate : candidates_) {
                auto &current_best = best_candidate[candidate.cell];
                if (candidate.probability > current_best.probability) {
                    current_best = candidate;
                }
            }

            for (const auto &[cell, candidate] : best_candidate) {
                const auto &record = records_[cell.record_index];
                const auto original_value = record.values.at(cell.attribute);

                RepairResult result;
                result.cell = cell;
                result.original_value = original_value;
                result.repaired_value = candidate.value;
                result.confidence = candidate.probability;
                result.applied = candidate.probability >= repair_threshold_;

                results_.push_back(result);
            }

            for (const auto &result : results_) {
                if (result.applied) {
                    records_[result.cell.record_index].values[result.cell.attribute] =
                        result.repaired_value;
                }
            }
        }

        [[nodiscard]] GroupKey build_group_key(const Record &record,
                                               const std::vector<std::string> &determinants) const {
            std::string key;
            for (const auto &attribute : determinants) {
                auto iter = record.values.find(attribute);
                if (iter != record.values.end()) {
                    key.append(attribute);
                    key.push_back('=');
                    key.append(iter->second);
                    key.push_back('|');
                }
            }
            return key;
        }

        double functional_dependency_support(const FunctionalDependency &fd,
                                             const GroupKey &key,
                                             const std::string &value) const {
            std::size_t count_matching = 0;
            std::size_t count_total = 0;
            for (const auto &record : records_) {
                if (build_group_key(record, fd.determinants) == key) {
                    ++count_total;
                    auto iter = record.values.find(fd.dependent);
                    if (iter != record.values.end() && iter->second == value) {
                        ++count_matching;
                    }
                }
            }
            if (count_total == 0) {
                return 0.0;
            }
            return static_cast<double>(count_matching) / static_cast<double>(count_total);
        }

        static double normalized_levenshtein(const std::string &a, const std::string &b) {
            if (a.empty()) {
                return b.empty() ? 0.0 : 1.0;
            }
            if (b.empty()) {
                return 1.0;
            }

            const std::size_t len_a = a.size();
            const std::size_t len_b = b.size();
            std::vector<std::vector<std::size_t>> dp(len_a + 1, std::vector<std::size_t>(len_b + 1));

            for (std::size_t i = 0; i <= len_a; ++i) {
                dp[i][0] = i;
            }
            for (std::size_t j = 0; j <= len_b; ++j) {
                dp[0][j] = j;
            }

            for (std::size_t i = 1; i <= len_a; ++i) {
                for (std::size_t j = 1; j <= len_b; ++j) {
                    std::size_t cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
                    dp[i][j] = std::min({
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1,
                        dp[i - 1][j - 1] + cost
                    });
                }
            }

            double distance = static_cast<double>(dp[len_a][len_b]);
            double max_len = static_cast<double>(std::max(len_a, len_b));
            return distance / max_len;
        }

        static double logistic(double x) {
            return 1.0 / (1.0 + std::exp(-x));
        }

        double feature_weights(const std::string &feature) const {
            auto iter = manual_feature_weights_.find(feature);
            if (iter != manual_feature_weights_.end()) {
                return iter->second;
            }

            if (feature == "prior_frequency")
                return 1.5;
            if (feature.rfind("fd:", 0) == 0)
                return 2.0;
            if (feature == "edit_distance")
                return 1.0;
            return 1.0;
        }

        std::vector<Record> records_;
        std::vector<FunctionalDependency> functional_dependencies_;
        std::unordered_set<CellCoordinate, CellCoordinateHasher> dirty_cells_;
        std::vector<Candidate> candidates_;
        std::vector<RepairResult> results_;
        std::unordered_map<std::string, double> manual_feature_weights_;
        double repair_threshold_{0.6};
    };
}

#endif // Nott_REPAIR_HPP
