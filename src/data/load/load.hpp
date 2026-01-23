#ifndef Nott_LOAD_HPP
#define Nott_LOAD_HPP
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <charconv>
#include <vector>
#include <optional>
#include <regex>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <limits>
#include <system_error>
#include <iostream>

#include <torch/torch.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "../../core.hpp"
#include "types.hpp"
#include "../transform/format/format.hpp"
namespace Nott::Data::Load {
    namespace Details {
        inline std::string trim_copy(const std::string& value)
        {
            const auto not_space = [](unsigned char character) { return !std::isspace(character); };
            auto first = std::find_if(value.begin(), value.end(), not_space);
            auto last = std::find_if(value.rbegin(), value.rend(), not_space).base();
            if (first >= last) {
                return {};
            }
            std::string result(first, last);
            while (!result.empty() && (result.front() == '"' || result.front() == '\'')) {
                result.erase(result.begin());
            }
            while (!result.empty() && (result.back() == '"' || result.back() == '\'')) {
                result.pop_back();
            }
            return result;
        }

        inline std::vector<std::string> split_csv_line(const std::string& line, char delimiter = ',')
        {
            std::vector<std::string> tokens;
            tokens.reserve(16);

            std::string cur;
            cur.reserve(128);
            bool in_quote = false;

            for (size_t i = 0; i < line.size(); ++i) {
                char c = line[i];

                if (c == '"') {
                    // handle escaped double quotes ("")
                    if (in_quote && i + 1 < line.size() && line[i + 1] == '"') {
                        cur.push_back('"');
                        ++i; // skip the escaped quote
                    } else {
                        in_quote = !in_quote;
                    }
                    continue;
                }

                if (!in_quote && c == delimiter) {
                    tokens.emplace_back(trim_copy(cur));
                    cur.clear();
                    continue;
                }

                cur.push_back(c);
            }

            // push last token
            tokens.emplace_back(trim_copy(cur));
            return tokens;
        }

        inline std::string strip_utf8_bom(std::string value)
        {
            constexpr unsigned char bom[] = {0xEF, 0xBB, 0xBF};
            if (value.size() >= 3 &&
                static_cast<unsigned char>(value[0]) == bom[0] &&
                static_cast<unsigned char>(value[1]) == bom[1] &&
                static_cast<unsigned char>(value[2]) == bom[2]) {
                value.erase(0, 3);
                }
            return value;
        }


        inline std::unordered_map<std::string, std::size_t> header_index_map(const std::vector<std::string>& header)
        {
            std::unordered_map<std::string, std::size_t> mapping;
            for (std::size_t index = 0; index < header.size(); ++index) {
                mapping[strip_utf8_bom(trim_copy(header[index]))] = index;
            }
            return mapping;
        }


        inline std::filesystem::path resolve_ptbxl_record_base(const std::filesystem::path& root,
                                                                const std::filesystem::path& preferred_records_dir,
                                                                const std::string& relative_field)
        {
            // relative_field is typically like "records500/21000/21837_hr" or "records100/00000/00001_lr"
            // We must return the *base* path without extension so that ".hea"/".dat" can be appended.
            auto trimmed = trim_copy(relative_field);
            if (trimmed.empty()) {
                return {};
            }

            auto ensure_base = [](std::filesystem::path p) {
                if (p.has_extension()) {
                    p.replace_extension(); // drop .hea/.dat if present
                }
                return p.lexically_normal();
            };

            auto base_exists = [](const std::filesystem::path& base) {
                return std::filesystem::exists(base.string() + ".hea") || std::filesystem::exists(base.string() + ".dat");
            };

            std::filesystem::path candidate(trimmed);

            // Normalize known directory aliases (e.g., '100' -> 'records100')
            auto normalize_dir_alias = [](const std::filesystem::path& p)->std::filesystem::path {
                auto parts = p;
                std::vector<std::string> items;
                for (auto &comp : parts) items.push_back(comp.string());
                for (auto &s : items) {
                    std::string t = s; for (auto &c : t) c = std::tolower(static_cast<unsigned char>(c));
                    if (t == "100") s = "records100"; else if (t == "500") s = "records500";
                    else if (t == "record100") s = "records100"; else if (t == "record500") s = "records500";
                }
                std::filesystem::path out; for (auto &s : items) out /= s; return out;
            };
            candidate = normalize_dir_alias(candidate);

            // If it's relative, first try root/<relative_field>
            if (candidate.is_relative()) {
                std::filesystem::path try1 = ensure_base(root / candidate);
                if (base_exists(try1)) {
                    return try1;
                }

                // If user passed only the filename part (e.g., "00001_lr"), prepend preferred_records_dir
                std::filesystem::path try2 = ensure_base(root / preferred_records_dir.filename() / candidate);
                if (base_exists(try2)) {
                    return try2;
                }

                // If still missing the subfolder layer (e.g., "records100/00001_lr"), infer directory from numeric prefix
                const auto stem = candidate.stem().string(); // e.g., "00001_lr"
                // extract leading integer
                std::size_t pos = 0;
                while (pos < stem.size() && std::isdigit(static_cast<unsigned char>(stem[pos]))) { ++pos; }
                std::filesystem::path try3;
                if (pos > 0) {
                    try {
                        int id = std::stoi(stem.substr(0, pos));
                        int bucket = (id / 1000) * 1000; // group folders by thousands
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "%05d", bucket);
                        try3 = ensure_base(root / preferred_records_dir.filename() / buf / stem);
                        if (base_exists(try3)) {
                            return try3;
                        }
                    } catch (...) {
                        // swallow and fall through to directory scan
                    }
                }

                // Last resort: scan immediate subdirs of preferred_records_dir for matching stem
                try {
                    const auto base_dir = root / preferred_records_dir.filename();
                    if (std::filesystem::exists(base_dir) && std::filesystem::is_directory(base_dir)) {
                        for (auto const& entry : std::filesystem::directory_iterator(base_dir)) {
                            if (!entry.is_directory()) continue;
                            std::filesystem::path probe = ensure_base(entry.path() / candidate.stem());
                            if (base_exists(probe)) {
                                return probe;
                            }
                        }
                    }
                } catch (...) {
                    // ignore errors from directory iteration and fall through
                }

                // As a final fallback, return the normalized guess under root/<relative_field> without extension
                return ensure_base(root / candidate);
            } else {
                // Absolute path given. Just normalize and strip extension.
                std::filesystem::path abs = ensure_base(candidate);
                if (base_exists(abs)) return abs;
                // Also try relative to root to be forgiving
                std::filesystem::path alt = ensure_base(root / candidate.relative_path());
                return base_exists(alt) ? alt : abs;
            }
        }



        inline std::optional<double> parse_float(const std::string& token)
        {
            const auto trimmed = trim_copy(token);
            if (trimmed.empty()) return std::nullopt;

            // fast, locale-independent parsing
            double value{};
            auto first = trimmed.data();
            auto last = trimmed.data() + trimmed.size();
            auto result = std::from_chars(first, last, value);

            if (result.ec == std::errc() && result.ptr == last) {
                return value;
            }

            // fallback to stod if from_chars didn't parse (some implementations had partial double support)
            try {
                std::size_t processed = 0;
                const double v = std::stod(trimmed, &processed);
                if (processed == 0) return std::nullopt;
                return v;
            } catch (...) {
                return std::nullopt;
            }
        }

        inline std::string lowercase(std::string value)
        {
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return value;
        }

        inline std::vector<std::size_t> resolve_csv_indices(const std::vector<std::string>& header,
                                                            const Type::CSVParameters& parameters)
        {
            std::vector<std::size_t> indices;
            if (parameters.columns.empty()) {
                indices.resize(header.size());
                std::iota(indices.begin(), indices.end(), 0);
                return indices;
            }

            const auto mapping = header_index_map(header);
            for (const auto& column : parameters.columns) {
                auto it = mapping.find(trim_copy(column));
                if (it == mapping.end()) {
                    throw std::runtime_error("CSV column not found: " + column);
                }
                indices.push_back(it->second);
            }
            return indices;
        }

        inline std::vector<std::size_t> resolve_csv_indices(std::size_t column_count,
                                                            const Type::CSVParameters& parameters)
        {
            std::vector<std::size_t> indices;
            if (parameters.columns.empty()) {
                indices.resize(column_count);
                std::iota(indices.begin(), indices.end(), 0);
                return indices;
            }

            indices.reserve(parameters.columns.size());
            for (const auto& column : parameters.columns) {
                std::size_t index = 0;
                auto [ptr, ec] = std::from_chars(column.data(), column.data() + column.size(), index);
                if (ec != std::errc() || ptr != column.data() + column.size()) {
                    throw std::runtime_error("CSV descriptor requested column '" + column + "' without header support");
                }
                if (index >= column_count) {
                    throw std::runtime_error("CSV column index out of range: " + column);
                }
                indices.push_back(index);
            }
            return indices;
        }

        inline torch::Tensor load_csv_tensor(const std::filesystem::path& root,
                                             const Type::CSV& descriptor)
        {
            const auto file_path = root / descriptor.file;
            std::ifstream file(file_path);
            if (!file) {
                throw std::runtime_error("Failed to open CSV file: " + file_path.string());
            }

            std::vector<std::size_t> column_indices;
            std::vector<float> buffer;
            buffer.reserve(1024);
            std::size_t column_count = 0;

            auto finalise_tensor = [&](std::size_t rows) {
                if (rows == 0 || column_indices.empty()) {
                    return torch::empty({0, 0}, torch::kFloat32);
                }
                auto tensor = torch::from_blob(buffer.data(),
                                               {static_cast<long>(rows), static_cast<long>(column_indices.size())},
                                               torch::TensorOptions().dtype(torch::kFloat32))
                                  .clone();
                return tensor;
            };

            std::string line;
            std::size_t rows = 0;

            if (descriptor.parameters.has_header) {
                if (!std::getline(file, line)) {
                    return torch::empty({0, 0}, torch::kFloat32);
                }
                line = strip_utf8_bom(line);
                auto header_tokens = split_csv_line(line, descriptor.parameters.delimiter);
                column_count = header_tokens.size();
                column_indices = resolve_csv_indices(header_tokens, descriptor.parameters);
            }

            while (std::getline(file, line)) {
                if (line.empty()) {
                    continue;
                }

                auto tokens = split_csv_line(line, descriptor.parameters.delimiter);
                if (!descriptor.parameters.has_header) {
                    if (column_indices.empty()) {
                        column_count = tokens.size();
                        column_indices = resolve_csv_indices(column_count, descriptor.parameters);
                    }
                }

                if (tokens.size() < column_indices.size()) {
                    continue;
                }

                for (const auto index : column_indices) {
                    if (index >= tokens.size()) {
                        throw std::runtime_error("CSV column index out of range for row " + std::to_string(rows));
                    }
                    const auto maybe_value = parse_float(tokens[index]);
                    if (!maybe_value.has_value()) {
                        throw std::runtime_error("CSV row contains non-numeric value at column " + std::to_string(index));
                    }
                    buffer.push_back(static_cast<float>(*maybe_value));
                }

                ++rows;
            }

            if (rows == 0) {
                return torch::empty({0, static_cast<long>(column_indices.size())}, torch::kFloat32);
            }

            return finalise_tensor(rows);
        }

        inline std::string normalise_line(std::string value, const Type::TextParameters& parameters)
        {
            if (parameters.trim_whitespace) {
                value = trim_copy(value);
            }
            if (parameters.lowercase) {
                value = lowercase(value);
            }
            return value;
        }

        inline torch::Tensor load_text_tensor(const std::filesystem::path& root,
                                              const Type::Text& descriptor)
        {
            const auto file_path = root / descriptor.file;
            std::ifstream file(file_path);
            if (!file) {
                throw std::runtime_error("Failed to open text file: " + file_path.string());
            }

            std::vector<std::string> lines;
            lines.reserve(1024);
            std::size_t max_length = 0;
            std::string line;
            while (std::getline(file, line)) {
                auto processed = normalise_line(line, descriptor.parameters);
                if (processed.empty() && !descriptor.parameters.keep_empty_lines) {
                    continue;
                }
                max_length = std::max(max_length, processed.size());
                lines.push_back(std::move(processed));
            }

            if (lines.empty()) {
                return torch::empty({0, 0}, torch::kInt64);
            }

            auto tensor = torch::zeros({static_cast<long>(lines.size()), static_cast<long>(max_length)}, torch::kInt64);
            auto accessor = tensor.accessor<int64_t, 2>();
            for (std::size_t row = 0; row < lines.size(); ++row) {
                const auto& current = lines[row];
                for (std::size_t col = 0; col < current.size(); ++col) {
                    accessor[row][col] = static_cast<int64_t>(static_cast<unsigned char>(current[col]));
                }
            }
            return tensor;
        }

        inline std::size_t binary_type_width(Type::BinaryDataType type)
        {
            switch (type) {
                case Type::BinaryDataType::Int8:
                case Type::BinaryDataType::UInt8:
                    return 1;
                case Type::BinaryDataType::Int16:
                case Type::BinaryDataType::UInt16:
                    return 2;
                case Type::BinaryDataType::Int32:
                case Type::BinaryDataType::UInt32:
                case Type::BinaryDataType::Float32:
                    return 4;
                case Type::BinaryDataType::Float64:
                    return 8;
            }
            return 1;
        }

        inline torch::ScalarType binary_type_to_scalar(Type::BinaryDataType type)
        {
            switch (type) {
                case Type::BinaryDataType::Int8:
                    return torch::kInt8;
                case Type::BinaryDataType::UInt8:
                    return torch::kUInt8;
                case Type::BinaryDataType::Int16:
                    return torch::kInt16;
                case Type::BinaryDataType::UInt16:
                    return torch::kUInt16;
                case Type::BinaryDataType::Int32:
                    return torch::kInt32;
                case Type::BinaryDataType::UInt32:
                    return torch::kUInt32;
                case Type::BinaryDataType::Float32:
                    return torch::kFloat32;
                case Type::BinaryDataType::Float64:
                    return torch::kFloat64;
            }
            return torch::kUInt8;
        }

        inline void byteswap_scalar(void* data, std::size_t width)
        {
            auto* bytes = static_cast<std::uint8_t*>(data);
            for (std::size_t i = 0; i < width / 2; ++i) {
                std::swap(bytes[i], bytes[width - 1 - i]);
            }
        }

        inline void maybe_byteswap_buffer(std::vector<std::uint8_t>& buffer,
                                          std::size_t width,
                                          bool little_endian)
        {
            const bool host_little_endian = [] {
                constexpr std::uint16_t value = 0x0102;
                return *(reinterpret_cast<const std::uint8_t*>(&value) + 1) == 0x02;
            }();

            if (host_little_endian == little_endian || width == 1) {
                return;
            }

            for (std::size_t offset = 0; offset + width <= buffer.size(); offset += width) {
                byteswap_scalar(buffer.data() + offset, width);
            }
        }

        inline torch::Tensor load_binary_tensor(const std::filesystem::path& root,
                                                const Type::Binary& descriptor)
        {
            const auto file_path = root / descriptor.file;
            std::ifstream file(file_path, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Failed to open binary file: " + file_path.string());
            }

            file.seekg(0, std::ios::end);
            const auto file_size = static_cast<std::size_t>(file.tellg());
            file.seekg(0, std::ios::beg);

            if (file_size == 0) {
                return torch::empty({0}, binary_type_to_scalar(descriptor.parameters.type));
            }

            std::vector<std::uint8_t> buffer(file_size);
            if (!file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()))) {
                throw std::runtime_error("Failed to read binary file: " + file_path.string());
            }

            const auto scalar_width = binary_type_width(descriptor.parameters.type);
            if (scalar_width == 0 || file_size % scalar_width != 0) {
                throw std::runtime_error("Binary file size is not a multiple of scalar width: " + file_path.string());
            }

            maybe_byteswap_buffer(buffer, scalar_width, descriptor.parameters.little_endian);

            std::size_t record_size = descriptor.parameters.record_size == 0
                ? 1
                : descriptor.parameters.record_size;
            const std::size_t scalar_count = buffer.size() / scalar_width;
            if (scalar_count % record_size != 0) {
                throw std::runtime_error("Binary descriptor record size does not divide stream: " + file_path.string());
            }

            const auto sample_count = static_cast<long>(scalar_count / record_size);
            const auto options = torch::TensorOptions().dtype(binary_type_to_scalar(descriptor.parameters.type));
            auto tensor = torch::empty({sample_count, static_cast<long>(record_size)}, options);
            std::memcpy(tensor.data_ptr(), buffer.data(), buffer.size());
            return tensor;
        }

        template <typename Descriptor>
                struct ImageDescriptorTraits {
            static_assert(!std::is_same_v<Descriptor, Descriptor>,
                          "Image descriptor missing trait specialisation");
        };

        template <>
        struct ImageDescriptorTraits<Type::PNG> {
            inline static constexpr std::array<const char*, 1> extensions = {".png"};
        };

        template <>
        struct ImageDescriptorTraits<Type::JPEG> {
            inline static constexpr std::array<const char*, 2> extensions = {".jpeg", ".jpg"};
        };

        template <>
        struct ImageDescriptorTraits<Type::JPG> {
            inline static constexpr std::array<const char*, 2> extensions = {".jpg", ".jpeg"};
        };

        template <>
        struct ImageDescriptorTraits<Type::BMP> {
            inline static constexpr std::array<const char*, 1> extensions = {".bmp"};
        };

        template <>
        struct ImageDescriptorTraits<Type::TIFF> {
            inline static constexpr std::array<const char*, 2> extensions = {".tiff", ".tif"};
        };

        template <>
        struct ImageDescriptorTraits<Type::PPM> {
            inline static constexpr std::array<const char*, 1> extensions = {".ppm"};
        };

        template <>
        struct ImageDescriptorTraits<Type::PGM> {
            inline static constexpr std::array<const char*, 1> extensions = {".pgm"};
        };

        template <>
        struct ImageDescriptorTraits<Type::PBM> {
            inline static constexpr std::array<const char*, 1> extensions = {".pbm"};
        };

        template <typename Descriptor>
        inline bool has_allowed_image_extension(const std::filesystem::path& path)
        {

            auto ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            const auto& allowed = ImageDescriptorTraits<Descriptor>::extensions;
            return std::any_of(allowed.begin(), allowed.end(), [&](const char* candidate) {
                return ext == candidate;
            });
        }
        inline std::vector<std::string> normalised_path_parts(const std::filesystem::path& value)
        {
            std::vector<std::string> parts;
            auto normalised = value.lexically_normal();
            for (const auto& component : normalised) {
                const auto token = component.string();
                if (token.empty() || token == "." || token == "/" || token == "\\") {
                    continue;
                }
                parts.push_back(token);
            }
            return parts;
        }

        inline std::vector<std::filesystem::path> resolve_image_directory(const std::filesystem::path& root, const std::filesystem::path& requested) {
            namespace fs = std::filesystem;

            if (requested.empty()) {
                throw std::runtime_error("Image folder descriptor requires a directory name");
            }

            if (requested.is_absolute()) {
                if (fs::exists(requested) && fs::is_directory(requested)) {
                    return {requested};
                }
                throw std::runtime_error("Image folder not found: " + requested.string());
            }

            const fs::path direct = root / requested;
            if (fs::exists(direct) && fs::is_directory(direct)) {
                return {direct};
            }

            if (!fs::exists(root) || !fs::is_directory(root)) {
                throw std::runtime_error("Image folder root not found: " + root.string());
            }

            const auto requested_parts = normalised_path_parts(requested);
            if (requested_parts.empty()) {
                throw std::runtime_error("Image folder descriptor requires a valid directory suffix");
            }

            std::vector<fs::path> matches;

            for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied), end; it != end; ++it) {
                std::error_code ec;
                if (!it->is_directory(ec) || ec) {
                    continue;
                }

                const auto current_parts = normalised_path_parts(it->path());
                if (current_parts.size() < requested_parts.size()) {
                    continue;
                }

                bool HaveMatchs = true;
                for (std::size_t i = 0; i < requested_parts.size(); ++i) {
                    const auto& suffix_component = requested_parts[requested_parts.size() - 1 - i];
                    const auto& candidate_component = current_parts[current_parts.size() - 1 - i];
                    if (suffix_component != candidate_component) {
                        HaveMatchs = false;
                        break;
                    }
                }

                if (!HaveMatchs)
                    continue;


                matches.push_back(it->path());
            }

            if (!matches.empty()) {
                std::sort(matches.begin(), matches.end());
                matches.erase(std::unique(matches.begin(), matches.end()), matches.end());
                return matches;
            }

            throw std::runtime_error("Image folder not found under root '" + root.string() +
                                     "' for pattern '" + requested.string() + "'");
        }

        template <typename Descriptor>

        inline std::vector<std::filesystem::path> collect_image_files(const std::filesystem::path& directory,
                                                                      bool recursive)
        {
            namespace fs = std::filesystem;
            std::vector<fs::path> files;
            const auto add_if_supported = [&](const fs::path& candidate) {
                if (fs::is_regular_file(candidate) && has_allowed_image_extension<Descriptor>(candidate)) {
                    files.push_back(candidate);
                }
            };

            if (!fs::exists(directory)) {
                throw std::runtime_error("Image folder not found: " + directory.string());
            }

            if (recursive) {
                for (const auto& entry : fs::recursive_directory_iterator(directory)) {
                    add_if_supported(entry.path());
                }
            } else {
                for (const auto& entry : fs::directory_iterator(directory)) {
                    add_if_supported(entry.path());
                }
            }

            std::sort(files.begin(), files.end());
            return files;
        }

        template <typename Descriptor>
        inline torch::Tensor load_image_folder_tensor(const std::filesystem::path& root, const Descriptor& descriptor) {
            namespace fs = std::filesystem;
            const auto directories = resolve_image_directory(root, descriptor.directory);
            std::vector<fs::path> image_files;
            for (const auto& directory : directories) {
                auto files = collect_image_files<Descriptor>(directory, descriptor.parameters.recursive);
                image_files.insert(image_files.end(), files.begin(), files.end());
            }
            if (image_files.empty()) {
                throw std::runtime_error("Image folders contain no supported files for pattern '" + descriptor.directory + "' under root '" + root.string() + "'");
            }

            std::vector<torch::Tensor> samples;
            samples.reserve(image_files.size());

            const auto& image_params = descriptor.parameters;
            const auto load_flag = image_params.grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
            std::optional<std::pair<int, int>> expected_hw;
            int max_rows = 0;
            int max_cols = 0;
            int min_rows = std::numeric_limits<int>::max();
            int min_cols = std::numeric_limits<int>::max();
            const bool track_sample_sizes = image_params.normalize_size || image_params.size[0] != -1 || image_params.size[1] != -1;
            std::vector<std::pair<int, int>> sample_hw;
            if (track_sample_sizes) {
                sample_hw.reserve(image_files.size());
            }
            auto options = torch::TensorOptions().dtype(torch::kFloat32);

            struct ColorOrderResolution {
                std::array<int, 3> channel_map{}; // maps destination channels to BGR input indices
                std::string normalized;
            };

            auto resolve_color_order = [](const std::string& configured) {
                ColorOrderResolution result{};
                result.normalized = configured.empty() ? std::string{"RGB"} : configured;
                std::transform(result.normalized.begin(), result.normalized.end(), result.normalized.begin(),
                               [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

                if (result.normalized.size() != 3) {
                    throw std::runtime_error("Image color order must contain exactly the letters R, G, and B once each ("
                        "received '" + result.normalized + "').");
                }

                auto letter_index = [](char c) -> int {
                    if (c == 'R') return 0;
                    if (c == 'G') return 1;
                    if (c == 'B') return 2;
                    return -1;
                };
                auto bgr_source_index = [](char c) -> int {
                    switch (c) {
                        case 'B': return 0;
                        case 'G': return 1;
                        case 'R': return 2;
                        default: throw std::runtime_error("Unsupported image color order letter.");
                    }
                };

                std::array<bool, 3> seen_letters{false, false, false};
                for (std::size_t i = 0; i < result.normalized.size(); ++i) {
                    const char channel = result.normalized[i];
                    const int idx = letter_index(channel);
                    if (idx < 0 || seen_letters[idx]) {
                        throw std::runtime_error("Image color order must be a permutation of 'R', 'G', 'B' (received '"
                                                 + result.normalized + "').");
                    }
                    seen_letters[idx] = true;
                    result.channel_map[i] = bgr_source_index(channel);
                }
                return result;
            };

            const auto color_order = resolve_color_order(image_params.color_order);
            const bool needs_channel_reorder = !image_params.grayscale && color_order.channel_map != std::array<int, 3>{0, 1, 2};

            for (const auto& file_path : image_files) {
                cv::Mat image = cv::imread(file_path.string(), load_flag);
                if (image.empty()) {
                    throw std::runtime_error("Failed to decode image: " + file_path.string());
                }

                if (track_sample_sizes) {
                    if (image_params.normalize_size) {
                        max_rows = std::max(max_rows, image.rows);
                        max_cols = std::max(max_cols, image.cols);
                        min_rows = std::min(min_rows, image.rows);
                        min_cols = std::min(min_cols, image.cols);
                    }
                    sample_hw.emplace_back(image.rows, image.cols);
                } else {
                    if (!expected_hw.has_value()) {
                        expected_hw = {image.rows, image.cols};
                    } else if (image.rows != expected_hw->first || image.cols != expected_hw->second) {
                        throw std::runtime_error("Image dimensions mismatch in folder: " + file_path.string());
                    }
                }

                cv::Mat image_float;
                const double scale = image_params.normalize_colors ? (1.0 / 255.0) : 1.0;
                image.convertTo(image_float, CV_32F, scale);

                if (needs_channel_reorder) {
                    cv::Mat reordered(image_float.size(), image_float.type());
                    const int from_to[] = {
                        color_order.channel_map[0], 0,
                        color_order.channel_map[1], 1,
                        color_order.channel_map[2], 2
                    };
                    cv::mixChannels(&image_float, 1, &reordered, 1, from_to, 3);
                    image_float = reordered;
                }

                torch::Tensor tensor;
                if (descriptor.parameters.grayscale) {
                    tensor = torch::from_blob(image_float.data, {image_float.rows, image_float.cols}, options).clone();
                    if (descriptor.parameters.channels_first) {
                        tensor = tensor.unsqueeze(0);
                    } else {
                        tensor = tensor.unsqueeze(2);
                    }
                } else {
                    tensor = torch::from_blob(image_float.data, {image_float.rows, image_float.cols, 3}, options).clone();
                    if (descriptor.parameters.channels_first) {
                        tensor = tensor.permute({2, 0, 1});
                    }
                }

                samples.push_back(std::move(tensor));
            }
            if (track_sample_sizes && !samples.empty()) {
                std::vector<int> target_size;
                const bool normalize_size = image_params.normalize_size;
                if (image_params.size[0] !=-1 || image_params.size[1] !=-1) {
                    target_size = image_params.size;
                } else if (normalize_size) {
                    target_size = std::vector<int>{min_rows, min_cols};
                }

                if (target_size[0]>0 || target_size[1]>0) {
                    if (sample_hw.size() != samples.size()) {
                        throw std::runtime_error("Internal error: missing sample size metadata for resizing.");
                    }

                    auto to_channels_first = [&](const torch::Tensor& tensor) {
                        if (image_params.channels_first) {
                            return std::pair<torch::Tensor, bool>{tensor, false};
                        }
                        if (tensor.dim() != 3) {
                            throw std::runtime_error("Image resize expects 3D sample tensors when channels_last is set.");
                        }
                        return std::pair<torch::Tensor, bool>{tensor.permute({2, 0, 1}), true};
                    };

                    auto restore_layout = [&](torch::Tensor tensor, bool permuted) {
                        if (permuted) {
                            tensor = tensor.permute({1, 2, 0});
                        }
                        return tensor;
                    };

                    for (std::size_t i = 0; i < samples.size(); ++i) {
                        const auto [rows, cols] = sample_hw[i];
                        if (rows == (target_size)[0] && cols == (target_size)[1]) {
                            continue;
                        }


                        auto [working, was_permuted] = to_channels_first(samples[i]);
                        if (normalize_size) {
                            const int rows_to_remove = std::max(0, rows - (target_size)[0]);
                            const int cols_to_remove = std::max(0, cols - (target_size)[1]);

                            if (rows_to_remove > 0) {
                                working = working.narrow(1, 0, (target_size)[0]);
                            }
                            if (cols_to_remove > 0) {
                                working = working.narrow(2, 0, (target_size)[1]);
                            }

                            if (rows_to_remove > 0 || cols_to_remove > 0) {
                                std::cout << "[Nott::Load::Universal]-> normalize_size applied on sample " << i << " (" << rows << "x" << cols
                                          << " -> " << (target_size)[0] << "x" << (target_size)[1]
                                          << "): removed " << rows_to_remove << " rows and "
                                          << cols_to_remove << " columns\n";
                            }
                        } else {
                            const bool needs_downscale = rows > (target_size)[0] || cols > (target_size)[1];
                            const bool needs_upscale = rows < (target_size)[0] || cols < (target_size)[1];

                            Nott::Data::Transform::Format::Options::ScaleOptions options;
                            options.size = std::vector<int>{static_cast<int>((target_size)[0]), static_cast<int>((target_size)[1])};
                            options.interp = image_params.InterpolationMode;
                            if (needs_downscale && !needs_upscale) {
                                working = Nott::Data::Transform::Format::Downsample(working, options);
                            } else if (needs_upscale && !needs_downscale) {
                                working = Nott::Data::Transform::Format::Upsample(working, options);
                            } else if (needs_downscale) {
                                // Mixed dimensions larger/smaller: treat as downscale to avoid overshooting.
                                working = Nott::Data::Transform::Format::Downsample(working, options);
                            } else if (needs_upscale) {
                                working = Nott::Data::Transform::Format::Upsample(working, options);
                            }
                        }

                        samples[i] = restore_layout(working, was_permuted);
                    }
                }
            }
            auto batch = torch::stack(samples);

            return batch;
        }




        inline std::unordered_map<std::string, std::string> load_scp_superclass_map(
            const std::filesystem::path& scp_csv,
            const std::unordered_map<std::string, std::int64_t>& allowed_classes)
        {
            std::ifstream scp_file(scp_csv);
            if (!scp_file) {
                throw std::runtime_error("Failed to open scp_statements.csv at " + scp_csv.string());
            }

            std::string header_line;
            if (!std::getline(scp_file, header_line)) {
                throw std::runtime_error("scp_statements.csv is empty");
            }

            const auto header_tokens = split_csv_line(header_line);
            const auto indices = header_index_map(header_tokens);

            // Column names vary across mirrors; prefer 'scp_code', fall back to 'statement' or 'code'. Avoid 'description'.
            auto code_it = indices.find("scp_code");
            if (code_it == indices.end()) code_it = indices.find("statement");
            if (code_it == indices.end()) code_it = indices.find("code");
            if (code_it == indices.end()) code_it = indices.find("description");

            const auto diagnostic_flag_it = indices.find("diagnostic");
            const auto diagnostic_class_it = indices.find("diagnostic_class");

            if (code_it == indices.end() || diagnostic_flag_it == indices.end() || diagnostic_class_it == indices.end()) {
                throw std::runtime_error("scp_statements.csv missing required columns: scp_code (or statement/code/description), diagnostic, diagnostic_class");
            }

            std::unordered_map<std::string, std::string> scp_to_superclass;
            std::string row;
            while (std::getline(scp_file, row)) {
                if (row.empty()) continue;
                const auto fields = split_csv_line(row);
                if (fields.size() <= std::max({code_it->second, diagnostic_flag_it->second, diagnostic_class_it->second})) {
                    continue;
                }

                auto scp_code = trim_copy(fields[code_it->second]);
                auto diagnostic_flag = trim_copy(fields[diagnostic_flag_it->second]);
                auto diagnostic_class = trim_copy(fields[diagnostic_class_it->second]);

                if (diagnostic_flag != "1" || scp_code.empty()) continue;
                // Only keep superclasses we model
                if (allowed_classes.find(diagnostic_class) == allowed_classes.end()) continue;

                scp_to_superclass[scp_code] = diagnostic_class;
            }
            return scp_to_superclass;
        }


        inline std::unordered_map<std::string, float> extract_superclass_votes(
            const std::string& scp_codes_field,
            const std::unordered_map<std::string, std::string>& scp_to_superclass)

        {
            static const std::regex code_weight_regex(R"((["'])([^"']+)\1\s*:\s*([0-9]*\.?[0-9]+))");
            std::unordered_map<std::string, float> superclass_votes;

            // Fallback heuristic: infer superclass from common PTB-XL code patterns when mapping is missing.
            auto infer_superclass = [](const std::string& raw)->std::string {
                std::string s; s.reserve(raw.size());
                for (char c : raw) s.push_back(std::toupper(static_cast<unsigned char>(c)));
                auto contains = [&](const char* pat){ return s.find(pat) != std::string::npos; };
                auto starts_with = [&](const char* pat){ return s.rfind(pat, 0) == 0; };

                if (s == "NORM" || s == "NORMAL") return "NORM";

                // Myocardial infarction
                if (s == "MI" || contains("INFAR") || contains("MI")) return "MI";

                // ST/T changes
                if (s == "STTC" || starts_with("ST") || contains("STD") || contains("STE") || contains("NST")
                    || contains("TINV") || contains("TWA"))
                    return "STTC";

                // Conduction disturbances
                if (contains("BBB") || contains("AVB") || s == "LAFB" || s == "LAHB" || s == "LPFB"
                    || s == "IVCD" || s == "WPW" || contains("BIFASC") || contains("TRIFASC"))
                    return "CD";

                // Hypertrophy
                if (s == "HYP" || contains("LVH") || contains("RVH"))
                    return "HYP";

                return "";
            };

            // First pass: parse explicit weights from scp_codes
            bool any_match = false;
            for (std::sregex_iterator match(scp_codes_field.begin(), scp_codes_field.end(), code_weight_regex);
                 match != std::sregex_iterator();
                 ++match) {
                any_match = true;
                const auto scp_code_raw = (*match)[2].str();
                const auto weight_token = (*match)[3].str();

                // Try mapping via scp_statements, fallback to heuristic
                auto it = scp_to_superclass.find(scp_code_raw);
                if (it == scp_to_superclass.end()) {
                    std::string up = scp_code_raw; for (auto &c : up) c = std::toupper(static_cast<unsigned char>(c));
                    it = scp_to_superclass.find(up);
                }
                std::string superclass = (it != scp_to_superclass.end()) ? it->second : infer_superclass(scp_code_raw);
                if (superclass.empty()) continue;

                float w = 0.f;
                try {
                    w = std::stof(weight_token);
                } catch (...) {
                    w = 0.f;
                }
                // Treat zero/unknown as presence vote to avoid dropping rows
                if (!(w > 0.f)) w = 1.0f;

                superclass_votes[superclass] += w;
                 }

            // If regex failed entirely (e.g., different formatting), try a ultra-simple fallback:
            if (!any_match) {
                // Look for quoted tokens (keys) without requiring weights
                static const std::regex key_only(R"((["'])([^"']+)\1)");
                for (std::sregex_iterator m(scp_codes_field.begin(), scp_codes_field.end(), key_only);
                     m != std::sregex_iterator(); ++m) {
                    const auto scp_code_raw = (*m)[2].str();
                    auto it = scp_to_superclass.find(scp_code_raw);
                    if (it == scp_to_superclass.end()) {
                        std::string up = scp_code_raw; for (auto &c : up) c = std::toupper(static_cast<unsigned char>(c));
                        it = scp_to_superclass.find(up);
                    }
                    std::string superclass = (it != scp_to_superclass.end()) ? it->second : infer_superclass(scp_code_raw);
                    if (superclass.empty()) continue;
                    superclass_votes[superclass] += 1.0f;
                     }
            }

            return superclass_votes;
        }


        inline std::optional<std::int64_t> select_superclass_label(
            const std::unordered_map<std::string, float>& superclass_votes,
            const std::unordered_map<std::string, std::int64_t>& class_to_index)
        {
            if (superclass_votes.empty()) {
                return std::nullopt;
            }

            const auto dominant = std::max_element(
                superclass_votes.begin(), superclass_votes.end(),
                [](const auto& left, const auto& right) { return left.second < right.second; });

            const auto label_it = class_to_index.find(dominant->first);
            if (label_it == class_to_index.end()) {
                return std::nullopt;
            }

            return label_it->second;
        }

        inline std::optional<torch::Tensor> read_ptbxl_signal(const std::filesystem::path& base_path,
                                                              std::optional<std::int64_t>& expected_length,
                                                              bool normalise)
        {
            std::filesystem::path hea_path = base_path;
            hea_path += ".hea";
            std::filesystem::path dat_path = base_path;
            dat_path += ".dat";

            if (!std::filesystem::exists(hea_path) || !std::filesystem::exists(dat_path)) {
                return std::nullopt;
            }

            std::ifstream header_stream(hea_path);
            if (!header_stream) {
                return std::nullopt;
            }

            std::string wfdb_header;
            if (!std::getline(header_stream, wfdb_header)) {
                return std::nullopt;
            }

            std::istringstream header_tokens(wfdb_header);
            std::string record_name;
            std::int64_t signal_count{};
            double frequency{};
            std::int64_t sample_count{};
            header_tokens >> record_name >> signal_count >> frequency >> sample_count;

            if (signal_count != 12 || sample_count <= 0) {
                return std::nullopt;
            }

            if (!expected_length) {
                expected_length = sample_count;
            } else if (*expected_length != sample_count) {
                return std::nullopt;
            }

            const auto expected_bytes = static_cast<std::uintmax_t>(signal_count) *
                                        static_cast<std::uintmax_t>(sample_count) * sizeof(std::int16_t);
            std::error_code size_error;
            const auto dat_size = std::filesystem::file_size(dat_path, size_error);
            if (size_error || dat_size < expected_bytes) {
                return std::nullopt;
            }

            std::ifstream dat_stream(dat_path, std::ios::binary);
            if (!dat_stream) {
                return std::nullopt;
            }

            std::vector<std::int16_t> raw(static_cast<std::size_t>(signal_count) *
                                          static_cast<std::size_t>(sample_count));
            dat_stream.read(reinterpret_cast<char*>(raw.data()),
                            static_cast<std::streamsize>(raw.size() * sizeof(std::int16_t)));
            if (dat_stream.gcount() != static_cast<std::streamsize>(raw.size() * sizeof(std::int16_t))) {
                return std::nullopt;
            }

            auto tensor = torch::from_blob(raw.data(), {sample_count, signal_count},
                                           torch::TensorOptions().dtype(torch::kInt16))
                               .clone()
                               .to(torch::kFloat32)
                               .transpose(0, 1);

            if (normalise) {
                auto mean = tensor.mean(1, true);
                auto std = tensor.std(1, true, true);
                std = torch::where(std < 1e-6, torch::ones_like(std), std);
                tensor = (tensor - mean) / std;
            }

            return tensor;
        }


        inline std::filesystem::path resolve_cifar10_root(const std::string& root) {
            auto candidate = std::filesystem::path(root);
            if (!std::filesystem::exists(candidate)) {
                throw std::runtime_error("Provided CIFAR-10 root directory does not exist: " + root);
            }

            if (std::filesystem::is_directory(candidate / "cifar-10-batches-bin")) {
                candidate /= "cifar-10-batches-bin";
            }

            if (!std::filesystem::exists(candidate / "data_batch_1.bin")) {
                throw std::runtime_error(
                    "Unable to locate CIFAR-10 binary batches in the provided root directory: " + candidate.string());
            }

            return candidate;
        }

        struct CIFAR10RawBatch {
            torch::Tensor inputs;
            torch::Tensor targets;
        };

        inline CIFAR10RawBatch read_cifar10_batch(const std::filesystem::path& file_path) {
            constexpr int64_t kChannels = 3;
            constexpr int64_t kHeight = 32;
            constexpr int64_t kWidth = 32;
            constexpr int64_t kImageSize = kChannels * kHeight * kWidth;
            constexpr int64_t kRecordSize = kImageSize + 1; // label byte + image bytes

            std::ifstream file(file_path, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Failed to open CIFAR-10 batch file: " + file_path.string());
            }

            file.seekg(0, std::ios::end);
            const auto file_size = static_cast<std::size_t>(file.tellg());
            file.seekg(0, std::ios::beg);

            if (file_size % kRecordSize != 0) {
                throw std::runtime_error("Unexpected CIFAR-10 batch file size for: " + file_path.string());
            }

            const auto num_samples = static_cast<int64_t>(file_size / kRecordSize);
            std::vector<std::uint8_t> buffer(file_size);
            if (!file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()))) {
                throw std::runtime_error("Failed to read CIFAR-10 batch file: " + file_path.string());
            }

            auto inputs = torch::empty({num_samples, kChannels, kHeight, kWidth}, torch::kUInt8);
            auto targets = torch::empty({num_samples}, torch::kInt64);

            auto inputs_accessor = inputs.accessor<std::uint8_t, 4>();
            auto targets_accessor = targets.accessor<int64_t, 1>();

            for (int64_t sample_index = 0; sample_index < num_samples; ++sample_index) {
                const auto base_offset = static_cast<std::size_t>(sample_index) * kRecordSize;
                targets_accessor[sample_index] = static_cast<int64_t>(buffer[base_offset]);

                const auto* pixel_ptr = buffer.data() + base_offset + 1;
                for (int64_t channel = 0; channel < kChannels; ++channel) {
                    for (int64_t pixel = 0; pixel < kHeight * kWidth; ++pixel) {
                        const auto h = pixel / kWidth;
                        const auto w = pixel % kWidth;
                        inputs_accessor[sample_index][channel][h][w] = *(pixel_ptr++);
                    }
                }
            }

            return {inputs, targets};
        }

        inline std::pair<torch::Tensor, torch::Tensor> concatenate_batches(
            std::vector<CIFAR10RawBatch>& batches) {
            if (batches.empty()) {
                return {torch::empty({0, 3, 32, 32}), torch::empty({0}, torch::kLong)};
            }

            std::vector<torch::Tensor> inputs_list;
            std::vector<torch::Tensor> targets_list;
            inputs_list.reserve(batches.size());
            targets_list.reserve(batches.size());

            for (auto& batch : batches) {
                inputs_list.push_back(std::move(batch.inputs));
                targets_list.push_back(std::move(batch.targets));
            }

            auto inputs = torch::cat(inputs_list, 0);
            auto targets = torch::cat(targets_list, 0);
            return {inputs, targets};
        }

        template <class Tensor>
        Tensor apply_fraction(Tensor tensor, std::size_t count) {
            if (tensor.dim() == 0) {
                return tensor;
            }
            if (tensor.size(0) <= static_cast<int64_t>(count)) {
                return tensor;
            }
            const auto copy_count = static_cast<int64_t>(count);
            return tensor.narrow(0, 0, copy_count).clone();
        }

        inline torch::Tensor normalise_inputs(const torch::Tensor& tensor) {
            if (tensor.numel() == 0) {
                return tensor.to(torch::kFloat32);
            }
            return tensor.to(torch::kFloat32) / 255.0f;
        }
        inline std::filesystem::path resolve_mnist_root(const std::string& root) {
            const std::array<std::filesystem::path, 3> candidates = {
                std::filesystem::path(root),
                std::filesystem::path(root) / "MNIST",
                std::filesystem::path(root) / "MNIST" / "raw"
            };

            const std::array<const char*, 4> required_files = {
                "train-images-idx3-ubyte",
                "train-labels-idx1-ubyte",
                "t10k-images-idx3-ubyte",
                "t10k-labels-idx1-ubyte"
            };

            for (const auto& candidate : candidates) {
                if (!std::filesystem::exists(candidate)) {
                    continue;
                }

                const bool has_all_files = std::all_of(required_files.begin(), required_files.end(), [&](const char* file) {
                    return std::filesystem::exists(candidate / file);
                });

                if (has_all_files) {
                    return candidate;
                }
            }

            throw std::runtime_error("Unable to locate MNIST dataset in the provided root: " + root);
        }

        inline std::uint32_t read_big_endian_u32(std::ifstream& file, const std::filesystem::path& file_path, const char* context) {
            std::array<std::uint8_t, 4> buffer{};
            if (!file.read(reinterpret_cast<char*>(buffer.data()), 4)) {
                throw std::runtime_error("Failed to read " + std::string(context) + " from " + file_path.string());
            }

            return (static_cast<std::uint32_t>(buffer[0]) << 24U) |
                   (static_cast<std::uint32_t>(buffer[1]) << 16U) |
                   (static_cast<std::uint32_t>(buffer[2]) << 8U) |
                   static_cast<std::uint32_t>(buffer[3]);
        }

        inline torch::Tensor read_idx_images(const std::filesystem::path& file_path) {
            std::ifstream file(file_path, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Failed to open MNIST image file: " + file_path.string());
            }

            const auto magic = read_big_endian_u32(file, file_path, "magic number");
            if (magic != 2051) {
                throw std::runtime_error("Unexpected MNIST image file magic number in " + file_path.string());
            }

            const auto count = static_cast<int64_t>(read_big_endian_u32(file, file_path, "image count"));
            const auto rows = static_cast<int64_t>(read_big_endian_u32(file, file_path, "rows"));
            const auto cols = static_cast<int64_t>(read_big_endian_u32(file, file_path, "columns"));

            if (rows != 28 || cols != 28) {
                throw std::runtime_error("MNIST image dimensions must be 28x28 in file: " + file_path.string());
            }

            if (count < 0) {
                throw std::runtime_error("Negative image count encountered in file: " + file_path.string());
            }

            const auto expected_size = static_cast<std::size_t>(count * rows * cols);
            std::vector<std::uint8_t> buffer(expected_size);
            if (!file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()))) {
                throw std::runtime_error("Failed to read MNIST image payload from: " + file_path.string());
            }

            if (static_cast<std::size_t>(file.gcount()) != expected_size) {
                throw std::runtime_error("MNIST image file truncated: " + file_path.string());
            }

            auto tensor = torch::empty({count, 1, rows, cols}, torch::kUInt8);
            std::memcpy(tensor.data_ptr<std::uint8_t>(), buffer.data(), buffer.size());
            return tensor;
        }

        inline torch::Tensor read_idx_labels(const std::filesystem::path& file_path) {
            std::ifstream file(file_path, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Failed to open MNIST label file: " + file_path.string());
            }

            const auto magic = read_big_endian_u32(file, file_path, "magic number");
            if (magic != 2049) {
                throw std::runtime_error("Unexpected MNIST label file magic number in " + file_path.string());
            }

            const auto count = static_cast<int64_t>(read_big_endian_u32(file, file_path, "label count"));
            if (count < 0) {
                throw std::runtime_error("Negative label count encountered in file: " + file_path.string());
            }

            std::vector<std::uint8_t> buffer(static_cast<std::size_t>(count));
            if (!file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()))) {
                throw std::runtime_error("Failed to read MNIST label payload from: " + file_path.string());
            }

            if (static_cast<std::size_t>(file.gcount()) != buffer.size()) {
                throw std::runtime_error("MNIST label file truncated: " + file_path.string());
            }

            auto tensor = torch::empty({count}, torch::kInt64);
            auto accessor = tensor.accessor<int64_t, 1>();
            for (int64_t index = 0; index < count; ++index) {
                accessor[index] = static_cast<int64_t>(buffer[static_cast<std::size_t>(index)]);
            }

            return tensor;
        }
        struct ETThRawDataset {
            torch::Tensor inputs;
            torch::Tensor targets;
        };

        inline ETThRawDataset read_etth_csv(const std::filesystem::path& file_path) {
            if (!std::filesystem::exists(file_path)) {
                throw std::runtime_error("ETTh CSV file does not exist: " + file_path.string());
            }

            std::ifstream file(file_path);
            if (!file) {
                throw std::runtime_error("Failed to open ETTh CSV file: " + file_path.string());
            }

            std::string header_line;
            if (!std::getline(file, header_line)) {
                throw std::runtime_error("ETTh CSV file is empty: " + file_path.string());
            }

            header_line = strip_utf8_bom(header_line);
            const auto header_tokens = split_csv_line(header_line);
            if (header_tokens.size() < 3) {
                throw std::runtime_error("ETTh CSV header must contain at least timestamp, one feature, and target columns: " + file_path.string());
            }

            const std::size_t column_count = header_tokens.size();
            const std::size_t target_index = column_count - 1;
            const std::size_t feature_count = column_count - 2; // skip timestamp + target

            if (feature_count == 0) {
                throw std::runtime_error("ETTh CSV header does not expose any feature columns: " + file_path.string());
            }

            std::vector<float> feature_buffer;
            std::vector<float> target_buffer;
            feature_buffer.reserve(static_cast<std::size_t>(1024) * feature_count);
            target_buffer.reserve(1024);

            std::vector<float> row_features;
            row_features.reserve(feature_count);

            std::string line;
            while (std::getline(file, line)) {
                if (line.empty()) {
                    continue;
                }

                const auto tokens = split_csv_line(line);
                if (tokens.size() < column_count) {
                    continue;
                }

                row_features.clear();
                bool valid = true;
                float target_value = 0.0f;

                for (std::size_t index = 1; index < column_count; ++index) {
                    const auto maybe_value = parse_float(tokens[index]);
                    if (!maybe_value.has_value()) {
                        valid = false;
                        break;
                    }

                    const float value = static_cast<float>(*maybe_value);
                    if (index == target_index) {
                        target_value = value;
                    } else {
                        row_features.push_back(value);
                    }
                }

                if (!valid || row_features.size() != feature_count) {
                    continue;
                }

                feature_buffer.insert(feature_buffer.end(), row_features.begin(), row_features.end());
                target_buffer.push_back(target_value);
            }

            if (target_buffer.empty()) {
                throw std::runtime_error("ETTh CSV contained no parseable rows: " + file_path.string());
            }

            const auto sample_count = static_cast<int64_t>(target_buffer.size());
            if (feature_buffer.size() != static_cast<std::size_t>(sample_count) * feature_count) {
                throw std::runtime_error("ETTh CSV feature buffer size mismatch: " + file_path.string());
            }

            auto inputs = torch::from_blob(feature_buffer.data(),
                                           {sample_count, static_cast<int64_t>(feature_count)},
                                           torch::TensorOptions().dtype(torch::kFloat32))
                              .clone();
            auto targets = torch::from_blob(target_buffer.data(),
                                            {sample_count},
                                            torch::TensorOptions().dtype(torch::kFloat32))
                               .clone();

            return {std::move(inputs), std::move(targets)};
        }
    }

    template <bool BufferVRAM = false, class DevicePolicyT = Core::DevicePolicy<BufferVRAM>>
    [[nodiscard]] inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    CIFAR10(const std::string& root, float train_fraction = 1.0f, float test_fraction = 1.0f, bool normalise = true) {
        const auto dataset_root = Details::resolve_cifar10_root(root);

        const std::array<const char*, 5> train_files = {
            "data_batch_1.bin",
            "data_batch_2.bin",
            "data_batch_3.bin",
            "data_batch_4.bin",
            "data_batch_5.bin"
        };

        std::vector<Details::CIFAR10RawBatch> train_batches;
        train_batches.reserve(train_files.size());
        for (const auto* filename : train_files) {
            train_batches.push_back(Details::read_cifar10_batch(dataset_root / filename));
        }

        auto test_batch = Details::read_cifar10_batch(dataset_root / "test_batch.bin");
        std::vector<Details::CIFAR10RawBatch> test_batches;
        test_batches.push_back(std::move(test_batch));

        auto [train_inputs, train_targets] = Details::concatenate_batches(train_batches);
        auto [test_inputs, test_targets] = Details::concatenate_batches(test_batches);

        const auto total_train = static_cast<std::size_t>(train_inputs.size(0));
        const auto total_test = static_cast<std::size_t>(test_inputs.size(0));

        const auto effective_train = std::clamp<std::size_t>(
            static_cast<std::size_t>(std::round(train_fraction * static_cast<float>(total_train))), 0, total_train);
        const auto effective_test = std::clamp<std::size_t>(
            static_cast<std::size_t>(std::round(test_fraction * static_cast<float>(total_test))), 0, total_test);

        train_inputs = Details::apply_fraction(std::move(train_inputs), effective_train);
        train_targets = Details::apply_fraction(std::move(train_targets), effective_train);
        test_inputs = Details::apply_fraction(std::move(test_inputs), effective_test);
        test_targets = Details::apply_fraction(std::move(test_targets), effective_test);

        if (normalise) {
            train_inputs = Details::normalise_inputs(train_inputs);
            test_inputs = Details::normalise_inputs(test_inputs);
        } else {
            train_inputs = train_inputs.to(torch::kFloat32);
            test_inputs = test_inputs.to(torch::kFloat32);
        }

        if constexpr (BufferVRAM) {
            const auto device = DevicePolicyT::select();
            train_inputs = train_inputs.to(device);
            train_targets = train_targets.to(device);
            test_inputs = test_inputs.to(device);
            test_targets = test_targets.to(device);
        }

        return {train_inputs, train_targets, test_inputs, test_targets};
    }

    template <bool BufferVRAM = false, class DevicePolicyT = Core::DevicePolicy<BufferVRAM>>
    [[nodiscard]] inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    ETTh(const std::string& csv_file, float train_fraction = 0.7f, float test_fraction = 0.2f, bool normalise = true) {
        auto [inputs, targets] = Details::read_etth_csv(csv_file);

        const auto total_samples = static_cast<std::int64_t>(inputs.size(0));
        const auto feature_size = inputs.size(1);

        train_fraction = std::max(train_fraction, 0.0f);
        test_fraction = std::max(test_fraction, 0.0f);

        auto train_count = std::clamp<std::int64_t>(
            static_cast<std::int64_t>(std::llround(train_fraction * static_cast<float>(total_samples))),
            std::int64_t{0},
            total_samples);

        const auto remaining_for_test = total_samples - train_count;
        auto test_count = std::clamp<std::int64_t>(
            static_cast<std::int64_t>(std::llround(test_fraction * static_cast<float>(total_samples))),
            std::int64_t{0},
            remaining_for_test);

        torch::Tensor train_inputs;
        torch::Tensor train_targets;
        torch::Tensor test_inputs;
        torch::Tensor test_targets;

        const auto float_opts = torch::TensorOptions().dtype(torch::kFloat32);

        if (train_count > 0) {
            train_inputs = inputs.narrow(0, 0, train_count).clone();
            train_targets = targets.narrow(0, 0, train_count).clone();
        } else {
            train_inputs = torch::empty({0, feature_size}, float_opts);
            train_targets = torch::empty({0}, float_opts);
        }

        if (test_count > 0) {
            test_inputs = inputs.narrow(0, train_count, test_count).clone();
            test_targets = targets.narrow(0, train_count, test_count).clone();
        } else {
            test_inputs = torch::empty({0, feature_size}, float_opts);
            test_targets = torch::empty({0}, float_opts);
        }

        if (normalise && train_count > 0) {
            auto mean = train_inputs.mean(0, true);
            auto std = train_inputs.std(0, false, true);
            std = torch::where(std < 1e-6, torch::ones_like(std), std);
            train_inputs = (train_inputs - mean) / std;
            if (test_count > 0) {
                test_inputs = (test_inputs - mean) / std;
            }
        }

        if constexpr (BufferVRAM) {
            const auto device = DevicePolicyT::select();
            train_inputs = train_inputs.to(device);
            train_targets = train_targets.to(device);
            test_inputs = test_inputs.to(device);
            test_targets = test_targets.to(device);
        }

        return {train_inputs, train_targets, test_inputs, test_targets};
    }


    template <typename Descriptor>
    inline torch::Tensor load_descriptor_tensor(const std::filesystem::path&, const Descriptor&) {
        static_assert(sizeof(Descriptor) == 0, "Universal loader received an unsupported descriptor type");
        return {};
    }

    template <>
    inline torch::Tensor load_descriptor_tensor<Type::CSV>(const std::filesystem::path& root, const Type::CSV& descriptor) {
        return Details::load_csv_tensor(root, descriptor);
    }

    template <>
    inline torch::Tensor load_descriptor_tensor<Type::Text>(const std::filesystem::path& root, const Type::Text& descriptor) {
        return Details::load_text_tensor(root, descriptor);
    }

    template <>
    inline torch::Tensor load_descriptor_tensor<Type::Binary>(const std::filesystem::path& root, const Type::Binary& descriptor) {
        return Details::load_binary_tensor(root, descriptor);
    }

    template <>
    inline torch::Tensor load_descriptor_tensor<Type::PNG>(const std::filesystem::path& root, const Type::PNG& descriptor) {
        return Details::load_image_folder_tensor(root, descriptor);
    }

    template <>
    inline torch::Tensor load_descriptor_tensor<Type::JPEG>(const std::filesystem::path& root, const Type::JPEG& descriptor) {
        return Details::load_image_folder_tensor(root, descriptor);
    }

    template <>
    inline torch::Tensor load_descriptor_tensor<Type::JPG>(const std::filesystem::path& root, const Type::JPG& descriptor) {
        return Details::load_image_folder_tensor(root, descriptor);
    }

    template <>
    inline torch::Tensor load_descriptor_tensor<Type::BMP>(const std::filesystem::path& root, const Type::BMP& descriptor) {
        return Details::load_image_folder_tensor(root, descriptor);
    }

    template <>
    inline torch::Tensor load_descriptor_tensor<Type::TIFF>(const std::filesystem::path& root, const Type::TIFF& descriptor) {
        return Details::load_image_folder_tensor(root, descriptor);
    }

    template <>
    inline torch::Tensor load_descriptor_tensor<Type::PPM>(const std::filesystem::path& root, const Type::PPM& descriptor) {
        return Details::load_image_folder_tensor(root, descriptor);
    }

    template <>
    inline torch::Tensor load_descriptor_tensor<Type::PGM>(const std::filesystem::path& root, const Type::PGM& descriptor) {
        return Details::load_image_folder_tensor(root, descriptor);
    }

    template <>
    inline torch::Tensor load_descriptor_tensor<Type::PBM>(const std::filesystem::path& root, const Type::PBM& descriptor) {
        return Details::load_image_folder_tensor(root, descriptor);
    }

    template <bool BufferVRAM = false, class DevicePolicyT = Core::DevicePolicy<BufferVRAM>, class InputDescriptorT, class TargetDescriptorT>
    [[nodiscard]] inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    Universal(const std::string& root, const InputDescriptorT& input_descriptor, const TargetDescriptorT& target_descriptor, Type::GlobalParameters global = {}) {
        namespace fs = std::filesystem;

        if (global.train_fraction < 0.0f || global.test_fraction < 0.0f) {
            throw std::runtime_error("Universal loader requires non-negative fractions");
        }

        if ((global.train_fraction + global.test_fraction) > 1.0f + 1e-5f) {
            throw std::runtime_error("Universal loader fractions must sum to <= 1");
        }

        const fs::path root_path(root);
        auto inputs_tensor = load_descriptor_tensor(root_path, input_descriptor);
        auto targets_tensor = load_descriptor_tensor(root_path, target_descriptor);

        if (inputs_tensor.dim() == 0 || targets_tensor.dim() == 0) {
            throw std::runtime_error("Universal loader descriptors must expose at least one dimension");
        }

        if (inputs_tensor.size(0) != targets_tensor.size(0)) {
            throw std::runtime_error("Universal loader descriptors produced mismatched sample counts");
        }

        if (global.shuffle && inputs_tensor.size(0) > 1) {
            auto permutation = torch::randperm(inputs_tensor.size(0), torch::TensorOptions().dtype(torch::kLong));
            inputs_tensor = inputs_tensor.index_select(0, permutation);
            targets_tensor = targets_tensor.index_select(0, permutation);
        }

        const auto total = static_cast<std::int64_t>(inputs_tensor.size(0));
        const auto train_fraction = std::clamp(global.train_fraction, 0.0f, 1.0f);
        const auto test_fraction = std::clamp(global.test_fraction, 0.0f, 1.0f);

        auto train_count = std::clamp<std::int64_t>(
            static_cast<std::int64_t>(std::llround(train_fraction * static_cast<float>(total))),
            std::int64_t{0},
            total);

        auto remaining = total - train_count;
        auto test_count = std::clamp<std::int64_t>(
            static_cast<std::int64_t>(std::llround(test_fraction * static_cast<float>(total))),
            std::int64_t{0},
            remaining);

        auto empty_input_shape = inputs_tensor.sizes().vec();
        empty_input_shape[0] = 0;
        auto empty_target_shape = targets_tensor.sizes().vec();
        empty_target_shape[0] = 0;

        torch::Tensor train_inputs;
        torch::Tensor train_targets;
        torch::Tensor test_inputs;
        torch::Tensor test_targets;

        if (train_count > 0) {
            train_inputs = inputs_tensor.narrow(0, 0, train_count).clone();
            train_targets = targets_tensor.narrow(0, 0, train_count).clone();
        } else {
            train_inputs = torch::empty(empty_input_shape, inputs_tensor.options());
            train_targets = torch::empty(empty_target_shape, targets_tensor.options());
        }

        if (test_count > 0) {
            test_inputs = inputs_tensor.narrow(0, train_count, test_count).clone();
            test_targets = targets_tensor.narrow(0, train_count, test_count).clone();
        } else {
            test_inputs = torch::empty(empty_input_shape, inputs_tensor.options());
            test_targets = torch::empty(empty_target_shape, targets_tensor.options());
        }

        if constexpr (BufferVRAM) {
            const auto device = DevicePolicyT::select();
            train_inputs = train_inputs.to(device);
            train_targets = train_targets.to(device);
            test_inputs = test_inputs.to(device);
            test_targets = test_targets.to(device);
        }

        return {train_inputs, train_targets, test_inputs, test_targets};
    }


    template <bool BufferVRAM = false, class DevicePolicyT = Core::DevicePolicy<BufferVRAM>>
    [[nodiscard]] inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    MNIST(const std::string& root, float train_fraction = 1.0f, float test_fraction = 1.0f, bool normalise = true) {
        const auto dataset_root = Details::resolve_mnist_root(root);

        auto train_inputs = Details::read_idx_images(dataset_root / "train-images-idx3-ubyte");
        auto train_targets = Details::read_idx_labels(dataset_root / "train-labels-idx1-ubyte");
        auto test_inputs = Details::read_idx_images(dataset_root / "t10k-images-idx3-ubyte");
        auto test_targets = Details::read_idx_labels(dataset_root / "t10k-labels-idx1-ubyte");

        if (train_inputs.size(0) != train_targets.size(0)) {
            throw std::runtime_error("MNIST training images and labels count mismatch");
        }

        if (test_inputs.size(0) != test_targets.size(0)) {
            throw std::runtime_error("MNIST test images and labels count mismatch");
        }

        const auto total_train = static_cast<std::size_t>(train_inputs.size(0));
        const auto total_test = static_cast<std::size_t>(test_inputs.size(0));

        const auto effective_train = std::clamp<std::size_t>(
            static_cast<std::size_t>(std::round(train_fraction * static_cast<float>(total_train))), 0, total_train);
        const auto effective_test = std::clamp<std::size_t>(
            static_cast<std::size_t>(std::round(test_fraction * static_cast<float>(total_test))), 0, total_test);

        train_inputs = Details::apply_fraction(std::move(train_inputs), effective_train);
        train_targets = Details::apply_fraction(std::move(train_targets), effective_train);
        test_inputs = Details::apply_fraction(std::move(test_inputs), effective_test);
        test_targets = Details::apply_fraction(std::move(test_targets), effective_test);

        if (normalise) {
            train_inputs = train_inputs.to(torch::kFloat32) / 255.0f;
            test_inputs = test_inputs.to(torch::kFloat32) / 255.0f;
        } else {
            train_inputs = train_inputs.to(torch::kFloat32);
            test_inputs = test_inputs.to(torch::kFloat32);
        }

        if constexpr (BufferVRAM) {
            const auto device = DevicePolicyT::select();
            train_inputs = train_inputs.to(device);
            train_targets = train_targets.to(device);
            test_inputs = test_inputs.to(device);
            test_targets = test_targets.to(device);
        }

        return {train_inputs, train_targets, test_inputs, test_targets};
    }

    template <bool BufferVRAM = false, class DevicePolicyT = Core::DevicePolicy<BufferVRAM>>
    [[nodiscard]] inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> PTBXL(const std::string& root, bool low_resolution, float train_fraction = 0.8f, bool normalise = true, bool multilabel = false, float multilabel_threshold = 0.0f) {
        namespace fs = std::filesystem;
        const std::filesystem::path root_path = std::filesystem::path(root);
        const std::filesystem::path database_csv = root_path / "ptbxl_database.csv";
        const std::filesystem::path scp_csv = root_path / "scp_statements.csv";
        const std::filesystem::path records_dir = (low_resolution ? (root_path / "records100") : (root_path / "records500"));

        static const std::unordered_map<std::string, std::int64_t> class_to_index{
            {"NORM", 0}, {"MI", 1}, {"STTC", 2}, {"CD", 3}, {"HYP", 4}
        };
        static const std::vector<std::string> class_order = {"NORM","MI","STTC","CD","HYP"};


        const auto scp_to_superclass = Details::load_scp_superclass_map(scp_csv, class_to_index);

        std::ifstream database_file(database_csv);
        if (!database_file) {
            throw std::runtime_error("Failed to open ptbxl_database.csv at " + database_csv.string());
        }

        std::string header_line;
        if (!std::getline(database_file, header_line)) {
            throw std::runtime_error("ptbxl_database.csv is empty");
        }
        const auto header_tokens = Details::split_csv_line(header_line);
        const auto header = Details::header_index_map(header_tokens);

        const auto filename_key = low_resolution ? "filename_lr" : "filename_hr";
        const auto filename_it = header.find(filename_key);
        const auto scp_codes_it = header.find("scp_codes");
        if (filename_it == header.end() || scp_codes_it == header.end()) {
            throw std::runtime_error("ptbxl_database.csv missing required columns: " + std::string(filename_key) + " and scp_codes");
        }

        std::vector<torch::Tensor> all_signals;
        std::vector<std::int64_t>  all_labels;
        std::vector<torch::Tensor> all_multilabels;
        std::optional<std::int64_t> expected_signal_length;
        std::int64_t rows_seen = 0;
        std::int64_t rows_with_filename = 0;
        std::int64_t rows_with_label = 0;
        std::int64_t bases_resolved = 0;
        std::int64_t signals_read = 0;


        std::string row;
        while (std::getline(database_file, row)) {
            ++rows_seen;
            if (row.empty()) continue;
            const auto fields = Details::split_csv_line(row);

            const auto needed_max = std::max(filename_it->second, scp_codes_it->second);
            if (fields.size() <= static_cast<std::size_t>(needed_max)) continue;

            rows_with_filename += 1;

            const auto base_path = Details::resolve_ptbxl_record_base(root_path, records_dir, fields[filename_it->second]);
            if (base_path.empty()) continue;
            bases_resolved += 1;


            const auto votes = Details::extract_superclass_votes(fields[scp_codes_it->second], scp_to_superclass);
            std::optional<std::int64_t> label;
            torch::Tensor multi;
            if (multilabel) {
                multi = torch::zeros({(long)class_order.size()}, torch::TensorOptions().dtype(torch::kFloat32));
                int pos_count = 0;
                for (const auto& kv : votes) {
                    // kv.first is superclass string, kv.second is vote weight
                    auto itc = class_to_index.find(kv.first);
                    if (itc == class_to_index.end()) continue;
                    if (kv.second > multilabel_threshold) {
                        multi.index_put_({itc->second}, 1.0f);
                        ++pos_count;
                    }
                }
                if (pos_count == 0) {
                    // If nothing passed threshold but we do have votes, take the argmax to avoid dropping the sample
                    if (!votes.empty()) {
                        auto best = std::max_element(votes.begin(), votes.end(),
                                                     [](auto& a, auto& b){ return a.second < b.second; });
                        auto itb = class_to_index.find(best->first);
                        if (itb != class_to_index.end()) {
                            multi.index_put_({itb->second}, 1.0f);
                            pos_count = 1;
                        }
                    }
                }
                if (pos_count == 0) continue; // truly no label
                rows_with_label += 1;
            } else {
                label = Details::select_superclass_label(votes, class_to_index);
                if (!label) continue;

                rows_with_label += 1;
            }


            auto signal = Details::read_ptbxl_signal(base_path, expected_signal_length, normalise);
            if (!signal) continue;
            signals_read += 1;

            all_signals.push_back(std::move(*signal));
            if (multilabel) {
                all_multilabels.push_back(std::move(multi));
            } else {
                all_labels.push_back(*label);
            }
        }

        if (all_signals.empty()) {
            std::ostringstream oss;
            oss << "No PTB-XL records matched. "
                << "root=" << root_path << " low_resolution=" << (low_resolution?"true":"false") << " records_dir=" << records_dir << "\n"
                << "rows_seen=" << rows_seen << " rows_with_filename=" << rows_with_filename
                << " rows_with_label=" << rows_with_label
                << " bases_resolved=" << bases_resolved
                << " signals_read=" << signals_read << "\n";
            throw std::runtime_error(oss.str());
        }

        // Stack -> shuffle -> split (ignore folds completely)
        auto inputs  = torch::stack(all_signals);
        torch::Tensor targets;
        if (multilabel) {
            if (all_multilabels.empty()) {
                throw std::runtime_error("PTB-XL multilabel: no labels constructed");
            }
            targets = torch::stack(all_multilabels).to(torch::kFloat32); // [N,5]
        } else {
            targets = torch::tensor(all_labels, torch::TensorOptions().dtype(torch::kInt64)); // [N]
        }

        if (inputs.size(0) > 1) {
            auto perm = torch::randperm(inputs.size(0), torch::TensorOptions().dtype(torch::kLong));
            inputs  = inputs.index_select(0, perm);
            targets = targets.index_select(0, perm);
        }

        train_fraction = std::clamp(train_fraction, 0.0f, 1.0f);
        const auto total = static_cast<std::int64_t>(inputs.size(0));
        auto train_count = static_cast<std::int64_t>(std::llround(train_fraction * static_cast<float>(total)));
        train_count = std::clamp<std::int64_t>(train_count, std::int64_t{1}, total);

        torch::Tensor train_inputs, train_targets, test_inputs, test_targets;
        if (train_count == total) {
            train_inputs  = inputs;
            train_targets = targets;
            test_inputs   = inputs.narrow(0, total, 0);
            test_targets  = targets.narrow(0, total, 0);
        } else {
            train_inputs  = inputs.narrow(0, 0, train_count);
            train_targets = targets.narrow(0, 0, train_count);
            test_inputs   = inputs.narrow(0, train_count, total - train_count);
            test_targets  = targets.narrow(0, train_count, total - train_count);
        }

        if (!normalise) {
            train_inputs = train_inputs.to(torch::kFloat32);
            test_inputs  = test_inputs.to(torch::kFloat32);
        }

        if constexpr (BufferVRAM) {
            const auto device = DevicePolicyT::select();
            train_inputs  = train_inputs.to(device);
            train_targets = train_targets.to(device);
            test_inputs   = test_inputs.to(device);
            test_targets  = test_targets.to(device);
        }

        return {train_inputs, train_targets, test_inputs, test_targets};
    }

}

#endif //Nott_LOAD_HPP