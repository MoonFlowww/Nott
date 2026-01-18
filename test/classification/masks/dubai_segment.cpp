#include "../../../include/Nott.h"
#include <array>
#include <iostream>
#include <tuple>
#include <vector>
#include <queue>
#include <limits>
#include <cmath>
#include <cstdint>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
namespace {
constexpr std::array<std::array<std::uint8_t, 3>, 6> kClassPalette{{
  std::array<std::uint8_t, 3>{60, 16, 152},
  std::array<std::uint8_t, 3>{110,193, 228},
  std::array<std::uint8_t, 3>{132, 41, 246},
  std::array<std::uint8_t, 3>{155, 155, 155},
  std::array<std::uint8_t, 3>{226, 169, 41},
  std::array<std::uint8_t, 3>{254, 221, 58},
}};
torch::Tensor ColorizeClassMap(const torch::Tensor& class_map) {
  TORCH_CHECK(class_map.dim() == 2, "Expected [H, W] class map");

  auto labels = class_map.to(torch::kLong).to(torch::kCPU).contiguous();
  const auto H = labels.size(0);
  const auto W = labels.size(1);

  torch::Tensor rgb = torch::zeros({H, W, 3}, torch::TensorOptions().dtype(torch::kUInt8));
  auto lacc = labels.accessor<int64_t, 2>();
  auto racc = rgb.accessor<std::uint8_t, 3>();

  for (int64_t y = 0; y < H; ++y) {
    for (int64_t x = 0; x < W; ++x) {
      auto cls = lacc[y][x];
      if (cls < 0 || cls >= static_cast<long>(kClassPalette.size())) {
        cls = 0;
      }
      const auto& color = kClassPalette[static_cast<std::size_t>(cls)];
      racc[y][x][0] = color[0];
      racc[y][x][1] = color[1];
      racc[y][x][2] = color[2];
    }
  }

  return rgb;
}
torch::Tensor ConvertRgbMasksToOneHot(const torch::Tensor& masks) {
  TORCH_CHECK(masks.dim() == 4, "Expected [B, 3, H, W]");
  TORCH_CHECK(masks.size(1) == 3, "RGB masks must have 3 channels");

  auto uint8_masks = masks.to(torch::kUInt8).contiguous();  // [B, 3, H, W]
  const auto B = uint8_masks.size(0);
  const auto H = uint8_masks.size(2);
  const auto W = uint8_masks.size(3);
  const auto C = static_cast<std::int64_t>(kClassPalette.size());

  // [B, 3, H, W] -> [B, H, W, 3] -> [N, 3]
  auto masks_flat = uint8_masks.permute({0, 2, 3, 1}).reshape({B * H * W, 3});

  std::vector<std::uint8_t> palette_vec;
  palette_vec.reserve(C * 3);
  for (const auto& rgb : kClassPalette) {
    palette_vec.push_back(rgb[0]);
    palette_vec.push_back(rgb[1]);
    palette_vec.push_back(rgb[2]);
  }

  auto palette = torch::tensor(palette_vec, torch::TensorOptions().dtype(torch::kUInt8)).view({C, 3}).to(uint8_masks.device());
  auto masks_i32   = masks_flat.to(torch::kInt32).unsqueeze(1); // [N, 1, 3]
  auto palette_i32 = palette.to(torch::kInt32).unsqueeze(0); // [1, C, 3]

  auto diff  = masks_i32 - palette_i32; // [N, C, 3], int32
  auto dist2 = diff.mul(diff).sum(-1); // [N, C],  int32

  auto min_idx = std::get<1>(dist2.min(1)); // [N]

  auto one_hot_flat = torch::nn::functional::one_hot(min_idx, C).to(torch::kFloat32); // [N, C]
  auto one_hot = one_hot_flat.view({B, H, W, C}).permute({0, 3, 1, 2}); // [B, C, H, W]

  return one_hot;
}
struct SuperpixelParams {
  int desired_superpixels;
  double nu;
  double sigma;
  double gamma;
  double phi;
  double alpha;
  double lambda;
  double Tc;
  double Ts;
  int max_iterations;
  int max_splits;
  double energy_tolerance;
};

struct SuperpixelSegmentationResult {
  cv::Mat labels; // CV_32S
  std::vector<cv::Point2f> centers;
  std::vector<cv::Mat> distance_maps; // One per center, CV_32F
};

cv::Mat ComputeColorGradientMagnitude(const cv::Mat& bgr32f) {
  // bgr32f: CV_32FC3, range [0,1]
  std::vector<cv::Mat> ch;
  cv::split(bgr32f, ch); // B,G,R

  cv::Mat grad2 = cv::Mat::zeros(bgr32f.size(), CV_32F);

  for (int c = 0; c < 3; ++c) {
    cv::Mat gx, gy;
    cv::Sobel(ch[c], gx, CV_32F, 1, 0, 3);
    cv::Sobel(ch[c], gy, CV_32F, 0, 1, 3);
    cv::Mat mag_c;
    cv::magnitude(gx, gy, mag_c);
    grad2 += mag_c.mul(mag_c); // sum of squared magnitudes
  }

  cv::Mat grad;
  cv::sqrt(grad2, grad);
  return grad;
}


void ComputeDensityAndSpeed(const cv::Mat& image, const SuperpixelParams& params, cv::Mat& density, cv::Mat& base_speed, double& total_density) {
  cv::Mat img32f;
  if (image.channels() == 3) {
    image.convertTo(img32f, CV_32F, 1.0 / 255.0);
  } else {
    cv::Mat tmp;
    image.convertTo(tmp, CV_32F, 1.0 / 255.0);
    cv::cvtColor(tmp, img32f, cv::COLOR_GRAY2BGR);
  }

  cv::Mat grad = ComputeColorGradientMagnitude(img32f);
  const double sigma = params.sigma > 0.0 ? params.sigma : 1.0;
  cv::Mat grad_smooth;
  cv::GaussianBlur(grad, grad_smooth, cv::Size(), sigma);

  const double nu    = params.nu    > 0.0 ? params.nu    : 1.0;
  const double gamma = params.gamma > 0.0 ? params.gamma : 1e-6;
  cv::Mat edge_measure = grad / (grad_smooth + gamma);

  cv::exp(edge_measure / nu, density);   // D(x) = exp(E/nu)
  base_speed = 1.0 / density;                  // V(x) = 1 / D(x)

  total_density = cv::sum(density)[0];
}


std::vector<cv::Point2f> InitializeCenters(const cv::Mat& image, int desired_superpixels, std::mt19937& rng) {
  const int rows = image.rows;
  const int cols = image.cols;
  const int total_pixels = rows * cols;
  const double step = std::sqrt(static_cast<double>(total_pixels) / static_cast<double>(desired_superpixels));

  std::uniform_real_distribution<float> jitter(-0.25f * static_cast<float>(step), 0.25f * static_cast<float>(step));

  std::vector<cv::Point2f> centers;
  for (double y = step / 2.0; y < rows; y += step) {
    for (double x = step / 2.0; x < cols; x += step) {
      centers.emplace_back(
        static_cast<float>(std::clamp(x + jitter(rng), 0.0, static_cast<double>(cols - 1))),
        static_cast<float>(std::clamp(y + jitter(rng), 0.0, static_cast<double>(rows - 1))));
    }
  }

  if (centers.size() > static_cast<std::size_t>(desired_superpixels)) {
    centers.resize(static_cast<std::size_t>(desired_superpixels));
  }

  return centers;
}

struct Node {
  int y;
  int x;
  float cost;
  bool operator<(const Node& other) const { return cost > other.cost; }
};

cv::Mat FastMarching(
  const cv::Mat& base_speed,
  const cv::Mat& density,
  const cv::Point& center,
  double avg_area,
  double alpha
) {
  const int rows = base_speed.rows;
  const int cols = base_speed.cols;

  cv::Mat dist(rows, cols, CV_32F, cv::Scalar(std::numeric_limits<float>::infinity()));
  cv::Mat finalized(rows, cols, CV_8U, cv::Scalar(0));
  std::priority_queue<Node> queue;

  const double sigma0 = std::max(1e-6, avg_area / std::max(alpha, 1e-6));
  double area_reached = 0.0;
  std::vector<std::pair<float, double>> cumulative_area_by_distance;
  cumulative_area_by_distance.reserve(static_cast<std::size_t>(rows * cols));

  auto area_at_distance = [&](float distance) {
    auto it = std::upper_bound(
      cumulative_area_by_distance.begin(),
      cumulative_area_by_distance.end(),
      distance,
      [](float d, const std::pair<float, double>& entry) {
        return d < entry.first;
      });
    if (it == cumulative_area_by_distance.begin()) {
      return 0.0;
    }
    return std::prev(it)->second;
  };

  dist.at<float>(center) = 0.0f;
  queue.push(Node{center.y, center.x, 0.0f});

  const std::array<std::pair<int, int>, 8> directions{{
    {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
  }};

  while (!queue.empty()) {
    const auto current = queue.top();
    queue.pop();

    if (current.cost > dist.at<float>(current.y, current.x) || finalized.at<std::uint8_t>(current.y, current.x)) {
      continue;
    }

    finalized.at<std::uint8_t>(current.y, current.x) = 1;
    area_reached += static_cast<double>(density.at<float>(current.y, current.x));
    cumulative_area_by_distance.emplace_back(current.cost, area_reached);

    for (const auto& dir : directions) {
      const int ny = current.y + dir.first;
      const int nx = current.x + dir.second;
      if (ny < 0 || ny >= rows || nx < 0 || nx >= cols) {
        continue;
      }

      if (finalized.at<std::uint8_t>(ny, nx)) {
        continue;
      }

      const float step = (dir.first == 0 || dir.second == 0) ? 1.0f : static_cast<float>(std::sqrt(2.0));
      const float base_v = std::max(base_speed.at<float>(ny, nx), 1e-6f);

      float tentative = current.cost + step / base_v;
      for (int refine = 0; refine < 2; ++refine) {
        const double base_area = area_at_distance(tentative);
        const double area_tentative = base_area + static_cast<double>(density.at<float>(ny, nx));
        const double delta = std::max(0.0, area_tentative - avg_area);
        const double gaussian = std::exp(-(delta * delta) / (2.0 * sigma0 * sigma0));
        const float v = static_cast<float>(base_v * gaussian);
        const float updated = current.cost + step / v;
        if (std::abs(updated - tentative) < 1e-6f) {
          tentative = updated;
          break;
        }
        tentative = updated;
      }

      float& best = dist.at<float>(ny, nx);
      if (tentative < best) {
        best = tentative;
        queue.push(Node{ny, nx, tentative});
      }
    }
  }

  return dist;
}

std::vector<cv::Mat> ComputeDistanceMaps(
  const cv::Mat& base_speed,
  const cv::Mat& density,
  const std::vector<cv::Point2f>& centers,
  double avg_area,
  double alpha
) {
  std::vector<cv::Mat> distance_maps;
  distance_maps.reserve(centers.size());

  for (std::size_t idx = 0; idx < centers.size(); ++idx) {
    cv::Point center_i(static_cast<int>(std::round(centers[idx].x)), static_cast<int>(std::round(centers[idx].y)));
    center_i.x = std::clamp(center_i.x, 0, base_speed.cols - 1);
    center_i.y = std::clamp(center_i.y, 0, base_speed.rows - 1);
    distance_maps.push_back(FastMarching(base_speed, density, center_i, avg_area, alpha));
  }

  return distance_maps;
}

void AssignLabels(const std::vector<cv::Mat>& distance_maps, cv::Mat& labels, cv::Mat& min_distance) {
  const int rows = distance_maps.front().rows;
  const int cols = distance_maps.front().cols;
  labels = cv::Mat(rows, cols, CV_32S, cv::Scalar(-1));
  min_distance = cv::Mat(rows, cols, CV_32F, cv::Scalar(std::numeric_limits<float>::infinity()));

  for (std::size_t idx = 0; idx < distance_maps.size(); ++idx) {
    for (int y = 0; y < rows; ++y) {
      const float* dist_ptr = distance_maps[idx].ptr<float>(y);
      float* min_ptr = min_distance.ptr<float>(y);
      int* label_ptr = labels.ptr<int>(y);
      for (int x = 0; x < cols; ++x) {
        const float d = dist_ptr[x];
        if (d < min_ptr[x]) {
          min_ptr[x] = d;
          label_ptr[x] = static_cast<int>(idx);
        }
      }
    }
  }
}

std::vector<double> ComputeAreas(const cv::Mat& labels, const cv::Mat& density, int superpixel_count) {
  std::vector<double> areas(static_cast<std::size_t>(superpixel_count), 0.0);
  for (int y = 0; y < labels.rows; ++y) {
    const int* lbl_ptr = labels.ptr<int>(y);
    const float* dens_ptr = density.ptr<float>(y);
    for (int x = 0; x < labels.cols; ++x) {
      const int lbl = lbl_ptr[x];
      if (lbl >= 0 && lbl < superpixel_count) {
        areas[static_cast<std::size_t>(lbl)] += dens_ptr[x];
      }
    }
  }
  return areas;
}

double ComputeEnergy(const cv::Mat& labels, const std::vector<cv::Mat>& distance_maps, const SuperpixelParams& params, const std::vector<double>& areas, double avg_area) {
  const int rows = labels.rows;
  const int cols = labels.cols;
  const int superpixel_count = static_cast<int>(areas.size());

  double E_image = 0.0;

  for (int l = 0; l < superpixel_count; ++l) {
    const cv::Mat& dist_l = distance_maps[static_cast<std::size_t>(l)];
    for (int y = 0; y < rows; ++y) {
      const int* lbl_ptr = labels.ptr<int>(y);
      const float* d_ptr  = dist_l.ptr<float>(y);
      for (int x = 0; x < cols; ++x) {
        if (lbl_ptr[x] != l) {
          continue;
        }
        const double d = static_cast<double>(d_ptr[x]);        // Dg(cl, x)
        const double w = std::exp(-d / std::max(1e-6, params.phi)); // Wx
        E_image += w * d * d;
      }
    }
  }

  double E_structure = 0.0;
  for (double area : areas) {
    const double diff = area - avg_area;
    E_structure += diff * diff;
  }

  return E_image + params.alpha * E_structure;
}


void RelocateCenters(std::vector<cv::Point2f>& centers, const cv::Mat& labels, const std::vector<cv::Mat>& distance_maps, const SuperpixelParams& params) {
  const int rows = labels.rows;
  const int cols = labels.cols;
  const int superpixel_count = static_cast<int>(centers.size());

  std::vector<cv::Point2f> new_centers(superpixel_count, cv::Point2f(0.0f, 0.0f));
  std::vector<double> weights(superpixel_count, 0.0);

  for (int l = 0; l < superpixel_count; ++l) {
    const cv::Mat& dist_l = distance_maps[static_cast<std::size_t>(l)];
    const cv::Point2f c = centers[static_cast<std::size_t>(l)];

    for (int y = 0; y < rows; ++y) {
      const int* lbl_ptr = labels.ptr<int>(y);
      const float* d_ptr = dist_l.ptr<float>(y);

      for (int x = 0; x < cols; ++x) {
        if (lbl_ptr[x] != l) {
          continue;
        }

        const float d = d_ptr[x]; // Dg(cl, x)
        if (d <= 0.0f || !std::isfinite(d)) {
          continue;
        }

        const float dx = static_cast<float>(x) - c.x;
        const float dy = static_cast<float>(y) - c.y;
        const float spatial = std::max(std::sqrt(dx * dx + dy * dy), 1e-6f);

        const float w_x = std::exp(-d / static_cast<float>(params.phi)); // Wx
        const float w   = w_x * d / spatial;

        new_centers[static_cast<std::size_t>(l)].x += w * static_cast<float>(x);
        new_centers[static_cast<std::size_t>(l)].y += w * static_cast<float>(y);
        weights[static_cast<std::size_t>(l)]      += w;
      }
    }
  }

  for (int l = 0; l < superpixel_count; ++l) {
    if (weights[static_cast<std::size_t>(l)] > 1e-6) {
      centers[static_cast<std::size_t>(l)].x =
        new_centers[static_cast<std::size_t>(l)].x / static_cast<float>(weights[static_cast<std::size_t>(l)]);
      centers[static_cast<std::size_t>(l)].y =
        new_centers[static_cast<std::size_t>(l)].y / static_cast<float>(weights[static_cast<std::size_t>(l)]);
    }
  }
}


void AttemptSplits(
  std::vector<cv::Point2f>& centers,
  const cv::Mat& labels,
  const std::vector<cv::Mat>& distance_maps,
  const cv::Mat& density,
  const cv::Mat& image_lab,
  const std::vector<double>& areas,
  double avg_area,
  const SuperpixelParams& params
) {
  const int superpixel_count = static_cast<int>(centers.size());
  if (superpixel_count >= params.max_splits + params.desired_superpixels) {
    return;
  }

  std::vector<cv::Point2f> new_centers;
  bool split_happened = false;

  struct RegionInfo {
    int index;
    double area;
    cv::Point2f center;
    cv::Vec2d principal_normal;
    std::vector<cv::Point> points;
  };
  std::vector<RegionInfo> regions;
  regions.reserve(superpixel_count);

  for (int i = 0; i < superpixel_count; ++i) {
    std::vector<cv::Point> points;
    points.reserve(labels.rows * labels.cols / std::max(1, superpixel_count));
    for (int y = 0; y < labels.rows; ++y) {
      const int* lbl_ptr = labels.ptr<int>(y);
      for (int x = 0; x < labels.cols; ++x) {
        if (lbl_ptr[x] == i) {
          points.emplace_back(x, y);
        }
      }
    }

    if (points.size() < 4) {
      continue;
    }

    cv::Scalar mean, stddev;
    std::vector<cv::Vec3f> colors;
    colors.reserve(points.size());
    for (const auto& p : points) {
      colors.push_back(image_lab.at<cv::Vec3f>(p));
    }
    cv::meanStdDev(colors, mean, stddev);
    const double color_std = (stddev[0] + stddev[1] + stddev[2]) / 3.0;

    cv::Mat cov = cv::Mat::zeros(2, 2, CV_64F);
    const cv::Point2f c = centers[static_cast<std::size_t>(i)];
    for (const auto& p : points) {
      const double gdist = distance_maps[static_cast<std::size_t>(i)].at<float>(p);
      const double dx = static_cast<double>(p.x) - c.x;
      const double dy = static_cast<double>(p.y) - c.y;
      const double norm = std::max(dx * dx + dy * dy, 1e-6);
      const double weight = (gdist * gdist) / norm;
      cov.at<double>(0, 0) += weight * dx * dx;
      cov.at<double>(0, 1) += weight * dx * dy;
      cov.at<double>(1, 0) += weight * dy * dx;
      cov.at<double>(1, 1) += weight * dy * dy;
    }

    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(cov, eigenvalues, eigenvectors);

    const double eig1 = eigenvalues.at<double>(0);
    const double eig2 = std::max(eigenvalues.at<double>(1), 1e-9);
    const double shape_ratio = eig1 / eig2;
    const double C_shape = std::max(shape_ratio, params.lambda * color_std);
    const double C_size = areas[static_cast<std::size_t>(i)] / std::max(avg_area, 1e-9);

    if (C_shape <= params.Tc && C_size <= params.Ts) {
      regions.push_back(RegionInfo{i, areas[static_cast<std::size_t>(i)], c,
        cv::Vec2d(eigenvectors.at<double>(0, 0), eigenvectors.at<double>(0, 1)),
        std::move(points)});
      continue;
    }

    const cv::Vec2d normal(eigenvectors.at<double>(0, 0), eigenvectors.at<double>(0, 1));
    std::vector<cv::Point> part1, part2;
    part1.reserve(points.size());
    part2.reserve(points.size());
    for (const auto& p : points) {
      cv::Vec2d diff(static_cast<double>(p.x) - c.x, static_cast<double>(p.y) - c.y);
      const double proj = diff.dot(normal);
      if (proj > 0) {
        part1.push_back(p);
      } else if (proj < 0) {
        part2.push_back(p);
      }
    }

    auto compute_new_center = [&](const std::vector<cv::Point>& pts) -> std::optional<cv::Point2f> {
      double wsum = 0.0;
      cv::Point2d accum(0.0, 0.0);
      for (const auto& p : pts) {
        const double gdist = distance_maps[static_cast<std::size_t>(i)].at<float>(p);
        const double dx = static_cast<double>(p.x) - c.x;
        const double dy = static_cast<double>(p.y) - c.y;
        const double spatial = std::max(std::sqrt(dx * dx + dy * dy), 1e-6);
        const double w = gdist / spatial;
        wsum += w;
        accum.x += w * p.x;
        accum.y += w * p.y;
      }
      if (wsum < 1e-6) {
        return std::nullopt;
      }
      return cv::Point2f(static_cast<float>(accum.x / wsum), static_cast<float>(accum.y / wsum));
    };

    if (!part1.empty()) {
      if (auto c1 = compute_new_center(part1)) {
        new_centers.push_back(*c1);
      }
    }
    if (!part2.empty()) {
      if (auto c2 = compute_new_center(part2)) {
        new_centers.push_back(*c2);
      }
    }

    split_happened = split_happened || !part1.empty() || !part2.empty();
  }

  if (!split_happened && static_cast<int>(centers.size()) < params.desired_superpixels) {
    std::sort(regions.begin(), regions.end(), [&](const RegionInfo& a, const RegionInfo& b) {
      return a.area > b.area;
    });

    auto partition_region = [&](const RegionInfo& region) {
      const cv::Point2f c = region.center;
      std::vector<cv::Point> part1, part2;
      part1.reserve(region.points.size());
      part2.reserve(region.points.size());
      for (const auto& p : region.points) {
        cv::Vec2d diff(static_cast<double>(p.x) - c.x, static_cast<double>(p.y) - c.y);
        const double proj = diff.dot(region.principal_normal);
        if (proj > 0) {
          part1.push_back(p);
        } else if (proj < 0) {
          part2.push_back(p);
        }
      }

      auto compute_new_center = [&](const std::vector<cv::Point>& pts) -> std::optional<cv::Point2f> {
        double wsum = 0.0;
        cv::Point2d accum(0.0, 0.0);
        for (const auto& p : pts) {
          const double gdist = distance_maps[static_cast<std::size_t>(region.index)].at<float>(p);
          const double dx = static_cast<double>(p.x) - c.x;
          const double dy = static_cast<double>(p.y) - c.y;
          const double spatial = std::max(std::sqrt(dx * dx + dy * dy), 1e-6);
          const double w = gdist / spatial;
          wsum += w;
          accum.x += w * p.x;
          accum.y += w * p.y;
        }
        if (wsum < 1e-6) {
          return std::nullopt;
        }
        return cv::Point2f(static_cast<float>(accum.x / wsum), static_cast<float>(accum.y / wsum));
      };

      if (!part1.empty()) {
        if (auto c1 = compute_new_center(part1)) {
          new_centers.push_back(*c1);
        }
      }
      if (!part2.empty()) {
        if (auto c2 = compute_new_center(part2)) {
          new_centers.push_back(*c2);
        }
      }
    };

    const int max_additional = std::min(params.max_splits,
                                        params.desired_superpixels - static_cast<int>(centers.size()));
    for (int k = 0; k < max_additional && k < static_cast<int>(regions.size()); ++k) {
      partition_region(regions[static_cast<std::size_t>(k)]);
      if (static_cast<int>(centers.size() + new_centers.size()) >= params.desired_superpixels + params.max_splits) {
        break;
      }
    }
  }

  for (const auto& nc : new_centers) {
    centers.push_back(nc);
    if (static_cast<int>(centers.size()) >= params.desired_superpixels + params.max_splits) {
      break;
    }
  }
}

SuperpixelSegmentationResult RunSuperpixelSegmentation(const cv::Mat& bgr_image, const SuperpixelParams& params) {
  cv::Mat density, base_speed;
  double total_density = 0.0;
  ComputeDensityAndSpeed(bgr_image, params, density, base_speed, total_density);

  std::mt19937 rng(42);
  auto centers = InitializeCenters(bgr_image, params.desired_superpixels, rng);
  std::vector<double> areas(centers.size(), total_density / std::max(1, params.desired_superpixels));

  cv::Mat labels, min_distance;
  cv::Mat lab_image;
  cv::cvtColor(bgr_image, lab_image, cv::COLOR_BGR2Lab);
  lab_image.convertTo(lab_image, CV_32F, 1.0 / 255.0);

  double prev_energy = std::numeric_limits<double>::infinity();
  const double energy_tolerance = std::max(params.energy_tolerance, 0.0);

  for (int iter = 0; iter < params.max_iterations; ++iter) {
    const double avg_area = total_density / std::max<std::size_t>(1, centers.size());
    auto distance_maps = ComputeDistanceMaps(base_speed, density, centers, avg_area, params.alpha);
    AssignLabels(distance_maps, labels, min_distance);

    areas = ComputeAreas(labels, density, static_cast<int>(centers.size()));
    RelocateCenters(centers, labels, distance_maps, params);
    AttemptSplits(centers, labels, distance_maps, density, lab_image, areas, avg_area, params);

    const double updated_avg_area = total_density / std::max<std::size_t>(1, centers.size());
    distance_maps = ComputeDistanceMaps(base_speed, density, centers, updated_avg_area, params.alpha);
    AssignLabels(distance_maps, labels, min_distance);
    areas = ComputeAreas(labels, density, static_cast<int>(centers.size()));
    const double energy = ComputeEnergy(labels, distance_maps, params, areas, updated_avg_area);

    if (std::isfinite(prev_energy)) {
      const double rel_change = std::abs(energy - prev_energy) / std::max(prev_energy, 1e-9);
      if (rel_change < energy_tolerance) {
        break;
      }
    }
    prev_energy = energy;
  }

  const double final_avg_area = total_density / std::max<std::size_t>(1, centers.size());
  auto final_distance_maps = ComputeDistanceMaps(base_speed, density, centers, final_avg_area, params.alpha);
  AssignLabels(final_distance_maps, labels, min_distance);

  return SuperpixelSegmentationResult{labels, centers, final_distance_maps};
}

torch::Tensor OriginalTile(torch::Tensor&x) {
  std::vector<torch::Tensor> rows; rows.reserve(3);
  for (int i=0; i<3; ++i) {
    auto inp1 = x[3*i+0];
    auto inp2 = x[3*i+1];
    auto inp3 = x[3*i+2];

    torch::Tensor row_inp = torch::cat({inp1, inp2, inp3}, 2);
    rows.push_back(row_inp);
  }
  return torch::cat(rows, 1);
}

torch::Tensor Cuts(torch::Tensor& x, std::vector<int> cuts, bool subcuts=false) {
  int H = x.size(1); int W = x.size(2);
  int h=H/cuts[0]; int w=W/cuts[1]; //cuts resolution

  std::size_t samples = subcuts ? (cuts[0]*cuts[1])+((cuts[0]-1)*(cuts[1]-1)) : (cuts[0]*cuts[1]); // x² + (x-1)²
  std::vector<torch::Tensor> out; out.reserve(samples);

  for(int row=0; row<cuts[0]; ++row) { // classic cuts
    for(int col=0; col<cuts[1]; ++col) {
      torch::Tensor patch = x.index({
        at::indexing::Slice(),
        at::indexing::Slice(h*row, h*(row+1)),
        at::indexing::Slice(w*col, w*(col+1))
      });

      out.push_back(patch);
    }
  }

  if (subcuts) {
    int hshift=h*0.5f; int wshift=w*0.5f;
    for(int row=0; row<cuts[0]-1; ++row) {
      for(int col=0; col<cuts[1]-1; ++col) {
        torch::Tensor patch = x.index({
          at::indexing::Slice(),
          at::indexing::Slice(h*row+hshift, h*(row+1)+hshift),
          at::indexing::Slice(w*col+wshift, w*(col+1)+wshift)
        });

        out.push_back(patch);
      }
    }
  }
  return torch::stack(out, 0);
}

std::vector<int> EstimCuts(int64_t h, int64_t w, int64_t target) {
  double rh = static_cast<double>(h) / static_cast<double>(target);
  double rw = static_cast<double>(w) / static_cast<double>(target);
  return {
    static_cast<int>(std::llround(rh)),
    static_cast<int>(std::llround(rw))
  };
}


struct SuperpixelSegmentationResult;
struct SuperpixelSegmentationResult;

struct SuperpixelParams;
struct SuperpixelSegmentationResult;
SuperpixelSegmentationResult RunSuperpixelSegmentation(
  const cv::Mat& bgr_image,
  const SuperpixelParams& params);


inline cv::Mat TensorCHWToBGRMat(const torch::Tensor& chw)
{
  TORCH_CHECK(chw.dim() == 3, "Expected [C,H,W] tensor");
  TORCH_CHECK(chw.size(0) == 3, "Expected C=3");

  torch::Tensor cpu = chw.detach().to(torch::kCPU).contiguous();

  const int C = static_cast<int>(cpu.size(0));
  const int H = static_cast<int>(cpu.size(1));
  const int W = static_cast<int>(cpu.size(2));
  (void)C;

  torch::Tensor hwc = cpu.permute({1, 2, 0}).contiguous(); // [H,W,3]

  cv::Mat rgb;
  if (hwc.scalar_type() == torch::kUInt8) {
    rgb = cv::Mat(H, W, CV_8UC3, hwc.data_ptr<uint8_t>()).clone();
  } else if (hwc.scalar_type() == torch::kFloat32) {
    torch::Tensor hwc_u8 = hwc.mul(255.0).clamp(0.0, 255.0).to(torch::kUInt8).contiguous();
    rgb = cv::Mat(H, W, CV_8UC3, hwc_u8.data_ptr<uint8_t>()).clone();
  } else if (hwc.scalar_type() == torch::kFloat64) {
    torch::Tensor hwc_f32 = hwc.to(torch::kFloat32);
    torch::Tensor hwc_u8  = hwc_f32.mul(255.0).clamp(0.0, 255.0).to(torch::kUInt8).contiguous();
    rgb = cv::Mat(H, W, CV_8UC3, hwc_u8.data_ptr<uint8_t>()).clone();
  } else {
    TORCH_CHECK(false, "Unsupported tensor dtype for image conversion");
  }

  cv::Mat bgr;
  cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR); // PyTorch usually stores RGB
  return bgr;
}
inline std::vector<SuperpixelSegmentationResult> RunSuperpixelsOnBatch(const torch::Tensor& images, const SuperpixelParams& params) {
  TORCH_CHECK(images.dim() == 4, "Expected [N,C,H,W] tensor");

  const int64_t N = images.size(0);
  const int64_t C = images.size(1);
  TORCH_CHECK(C == 3, "Expected 3 channels (RGB)");

  std::vector<SuperpixelSegmentationResult> results;
  results.reserve(static_cast<std::size_t>(N));

  for (int64_t n = 0; n < N; ++n) {
    // images[n] is [C,H,W]
    torch::Tensor sample_chw = images[n];

    cv::Mat bgr_img = TensorCHWToBGRMat(sample_chw);
    SuperpixelSegmentationResult res = RunSuperpixelSegmentation(bgr_img, params);
    results.emplace_back(std::move(res));
  }

  return results;
}

SuperpixelSegmentationResult RunSuperpixelSegmentation(
  const cv::Mat& bgr_image,
  const SuperpixelParams& params);

// ---- Helper: Tensor [C,H,W] -> BGR cv::Mat --------------------------------



inline torch::Tensor ComputeSuperpixelRGBStats(
  const torch::Tensor& input_chw,
  const cv::Mat& labels,
  int num_superpixels)
{
  TORCH_CHECK(input_chw.dim() == 3, "input_chw must be [3,H,W]");
  TORCH_CHECK(input_chw.size(0) == 3, "input_chw must have 3 channels");
  TORCH_CHECK(labels.type() == CV_32S, "labels must be CV_32S");

  const int H = static_cast<int>(input_chw.size(1));
  const int W = static_cast<int>(input_chw.size(2));
  TORCH_CHECK(labels.rows == H && labels.cols == W, "labels size must match input spatial size");

  torch::Tensor in_cpu = input_chw.detach().to(torch::kCPU).contiguous().to(torch::kFloat32);
  auto in = in_cpu.accessor<float,3>(); // [3,H,W]

  const int C = 3;
  const int S = num_superpixels;

  const int SC = S * C;

  std::vector<int64_t> count(SC, 0);
  std::vector<double> sum(SC, 0.0);
  std::vector<double> sum2(SC, 0.0);
  std::vector<double> sum3(SC, 0.0);
  std::vector<double> minv(SC, std::numeric_limits<double>::infinity());
  std::vector<double> maxv(SC, -std::numeric_limits<double>::infinity());

  for (int y = 0; y < H; ++y) {
    const int* lbl_ptr = labels.ptr<int>(y);
    for (int x = 0; x < W; ++x) {
      const int sid = lbl_ptr[x];
      if (sid < 0 || sid >= S) {
        continue;
      }
      for (int c = 0; c < C; ++c) {
        const double v = static_cast<double>(in[c][y][x]);
        const int idx = sid * C + c;

        count[idx] += 1;
        sum[idx]   += v;
        sum2[idx]  += v * v;
        sum3[idx]  += v * v * v;
        if (v < minv[idx]) minv[idx] = v;
        if (v > maxv[idx]) maxv[idx] = v;
      }
    }
  }

  torch::Tensor feats = torch::empty({S, 5 * C}, torch::TensorOptions().dtype(torch::kFloat32));
  auto F = feats.accessor<float,2>(); // [S, 15]

  const double eps = 1e-12;

  for (int s = 0; s < S; ++s) {
    for (int c = 0; c < C; ++c) {
      const int idx = s * C + c;
      const int64_t n = count[idx];

      double mean = 0.0;
      double std  = 0.0;
      double skew = 0.0;
      double mn   = 0.0;
      double mx   = 0.0;

      if (n > 0) {
        const double inv_n = 1.0 / static_cast<double>(n);
        const double m1 = sum[idx]  * inv_n;
        const double m2 = sum2[idx] * inv_n;
        const double m3 = sum3[idx] * inv_n;

        const double var = std::max(m2 - m1 * m1, 0.0);
        const double sigma = std::sqrt(var + eps);

        const double mu3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 * m1 * m1;
        const double skewness = mu3 / std::pow(sigma + eps, 3.0);

        mean = m1;
        std  = sigma;
        skew = skewness;
        mn   = minv[idx];
        mx   = maxv[idx];
      } else {
        // empty superpixel: set all stats to 0
        mean = std = skew = mn = mx = 0.0;
      }

      const int base = c * 5;
      F[s][base + 0] = static_cast<float>(mn);   // MIN
      F[s][base + 1] = static_cast<float>(mx);   // MAX
      F[s][base + 2] = static_cast<float>(mean); // AVG
      F[s][base + 3] = static_cast<float>(std);  // STD
      F[s][base + 4] = static_cast<float>(skew); // SKEW
    }
  }

  return feats; // [S, 15]
}
inline torch::Tensor ComputeSuperpixelTargets(
  const torch::Tensor& target_chw,
  const cv::Mat& labels,
  int num_superpixels)
{
  TORCH_CHECK(target_chw.dim() == 3, "target_chw must be [K,H,W]");
  TORCH_CHECK(labels.type() == CV_32S, "labels must be CV_32S");

  const int K = static_cast<int>(target_chw.size(0));
  const int H = static_cast<int>(target_chw.size(1));
  const int W = static_cast<int>(target_chw.size(2));

  TORCH_CHECK(labels.rows == H && labels.cols == W, "labels size must match target spatial size");

  torch::Tensor tgt_cpu = target_chw.detach().to(torch::kCPU).contiguous().to(torch::kFloat32);
  auto T = tgt_cpu.accessor<float,3>(); // [K,H,W]

  const int S = num_superpixels;
  std::vector<double> sum_class(S * K, 0.0);

  for (int y = 0; y < H; ++y) {
    const int* lbl_ptr = labels.ptr<int>(y);
    for (int x = 0; x < W; ++x) {
      const int sid = lbl_ptr[x];
      if (sid < 0 || sid >= S) {
        continue;
      }
      for (int k = 0; k < K; ++k) {
        const double v = static_cast<double>(T[k][y][x]);
        sum_class[sid * K + k] += v;
      }
    }
  }

  torch::Tensor y = torch::empty({S}, torch::TensorOptions().dtype(torch::kLong));
  auto y_data = y.data_ptr<int64_t>();

  for (int s = 0; s < S; ++s) {
    int best_k = 0;
    double best_val = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < K; ++k) {
      const double v = sum_class[s * K + k];
      if (v > best_val) {
        best_val = v;
        best_k = k;
      }
    }
    y_data[s] = static_cast<int64_t>(best_k); // argmax over classes for this superpixel
  }

  return y; // [S], labels in [0..K-1]
}
inline torch::Tensor BGRMatToTensorCHW(const cv::Mat& bgr)
{
  TORCH_CHECK(bgr.type() == CV_8UC3, "Expected CV_8UC3 image");
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  auto options = torch::TensorOptions().dtype(torch::kUInt8);
  torch::Tensor hwc = torch::from_blob(rgb.data, {rgb.rows, rgb.cols, 3}, options).clone();
  return hwc.permute({2, 0, 1}); // [3,H,W]
}

inline cv::Mat DrawVoronoiOverlay(const cv::Mat& bgr_image, const cv::Mat& labels)
{
  TORCH_CHECK(labels.type() == CV_32S, "labels must be CV_32S");
  cv::Mat overlay = bgr_image.clone();
  cv::Mat boundary = cv::Mat::zeros(labels.size(), CV_8UC1);

  const int rows = labels.rows;
  const int cols = labels.cols;
  for (int y = 0; y < rows; ++y) {
    const int* row = labels.ptr<int>(y);
    const int* prev = (y > 0) ? labels.ptr<int>(y - 1) : nullptr;
    const int* next = (y + 1 < rows) ? labels.ptr<int>(y + 1) : nullptr;
    uchar* bptr = boundary.ptr<uchar>(y);

    for (int x = 0; x < cols; ++x) {
      const int lbl = row[x];
      bool edge = false;

      if (x > 0 && row[x - 1] != lbl) edge = true;
      if (x + 1 < cols && row[x + 1] != lbl) edge = true;
      if (prev && prev[x] != lbl) edge = true;
      if (next && next[x] != lbl) edge = true;

      if (edge) {
        bptr[x] = 255;
      }
    }
  }

  overlay.setTo(cv::Scalar(0, 0, 255), boundary);
  return overlay;
}


struct SuperpixelDataset {
  torch::Tensor X; // features
  torch::Tensor y; // labels (int64)
};

inline SuperpixelDataset BuildSuperpixelDataset(
  const torch::Tensor& inputs,   // [N,3,H,W]
  const torch::Tensor& targets,  // [N,K,H,W]
  const SuperpixelParams& params)
{
  TORCH_CHECK(inputs.dim()  == 4, "inputs must be [N,3,H,W]");
  TORCH_CHECK(targets.dim() == 4, "targets must be [N,K,H,W]");
  TORCH_CHECK(inputs.size(0) == targets.size(0), "inputs and targets must have same N");
  TORCH_CHECK(inputs.size(2) == targets.size(2) &&
              inputs.size(3) == targets.size(3),
              "inputs and targets must have same H,W");
  TORCH_CHECK(inputs.size(1) == 3, "inputs must have 3 channels");

  const int64_t N = inputs.size(0);
  const int64_t K = targets.size(1);

  std::vector<torch::Tensor> feats_list;
  std::vector<torch::Tensor> labels_list;

  feats_list.reserve(static_cast<std::size_t>(N));
  labels_list.reserve(static_cast<std::size_t>(N));

  for (int64_t n = 0; n < N; ++n) {
    // Extract single sample [3,H,W] and [K,H,W]
    torch::Tensor x_chw = inputs[n];   // [3,H,W]
    torch::Tensor t_chw = targets[n];  // [K,H,W]

    // Run superpixel segmentation on input image
    cv::Mat bgr_img = TensorCHWToBGRMat(x_chw);
    SuperpixelSegmentationResult seg = RunSuperpixelSegmentation(bgr_img, params);

    const int num_superpixels = static_cast<int>(seg.centers.size());

    // 1) Features: MIN/MAX/AVG/STD/SKEW over R,G,B
    torch::Tensor X_i = ComputeSuperpixelRGBStats(x_chw, seg.labels, num_superpixels); // [S,15]

    // 2) Target: for each superpixel, argmax over 6 classes
    torch::Tensor y_i = ComputeSuperpixelTargets(t_chw, seg.labels, num_superpixels);  // [S]

    feats_list.push_back(X_i);
    labels_list.push_back(y_i);
  }

  torch::Tensor X = torch::cat(feats_list, 0);      // [total_S, 15]
  torch::Tensor y = torch::cat(labels_list, 0);     // [total_S]

  return SuperpixelDataset{X, y};
}


void savePic(torch::Tensor inp, const std::string path, const std::string kind) {
  torch::Tensor t = inp.detach().to(torch::kCPU);
  t = t.clamp(0, 255).to(torch::kU8);
  torch::Tensor t_hwc = t.permute({1, 2, 0}).contiguous();
  int H = t_hwc.size(0);
  int W = t_hwc.size(1);
  cv::Mat img(H, W, CV_8UC3, t_hwc.data_ptr<uint8_t>());
  cv::Mat img_bgr;
  cv::cvtColor(img, img_bgr, cv::COLOR_RGB2BGR);
  cv::imwrite(path+"/"+kind+"_raw.png", img_bgr);
  cv::GaussianBlur(img_bgr, img_bgr, cv::Size(), 0.83);
  cv::imwrite(path+"/"+kind+"_smooth.png", img_bgr);
}

inline SuperpixelDataset BuildSuperpixelDatasetForLimitedVoronoiCuts(
  const torch::Tensor& inputs,
  const torch::Tensor& targets,
  const SuperpixelParams& params,
  std::size_t max_cuts = 20) {

  TORCH_CHECK(inputs.size(0) == targets.size(0), "Inputs and targets must have the same batch size");

  const int64_t limited = std::min<int64_t>(inputs.size(0), static_cast<int64_t>(max_cuts));
  auto slice = torch::indexing::Slice(0, limited);

  torch::Tensor limited_inputs = inputs.index({slice});
  torch::Tensor limited_targets = targets.index({slice});

  return BuildSuperpixelDataset(limited_inputs, limited_targets, params);
}

}

int _main() {
  Nott::Model model("");
  std::vector<torch::Tensor> xs;
  std::vector<torch::Tensor> ys;

  for (int tile = 1; tile <= 1; ++tile) { // 9
    std::string path = "/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile " + std::to_string(tile);
    // cuts(3*3) -> Tile -> cuts(X*Y)
    auto [x1, y1, x2, y2] =
      Nott::Data::Load::Universal(path,
                                  Nott::Data::Type::JPG{"images", {.grayscale = false, .normalize_colors = false, .normalize_size = true, .color_order = "RGB"}},
                                  Nott::Data::Type::PNG{"masks",  {.normalize_colors = false, .normalize_size = true, .InterpolationMode = Nott::Data::Transform::Format::Options::InterpMode::Nearest, .color_order = "RGB"}},
                                  {.train_fraction = 1.f, .test_fraction = 0.f, .shuffle = false});

    x1 = OriginalTile(x1);
    Nott::Data::Check::Size(x1, ("X"+std::to_string(tile) + " Reconstructed Tile"));
    savePic(x1, path, "input");
    y1 = OriginalTile(y1);
    savePic(y1, path, "target");
    Nott::Data::Check::Size(x1, ("Y"+std::to_string(tile) + " Reconstructed Tile"));
    auto EstX= EstimCuts(x1.size(1), x1.size(2), 128);
    auto EstY= EstimCuts(y1.size(1), y1.size(2), 128);
    std::cout << "Estimator {X,Y}: " << EstX << " || n: " << EstX[0]*EstX[1] << " + " << (EstX[0]-1)*(EstX[1]-1) << std::endl;
    x1 = Cuts(x1, EstX, true); // Estim used to minimize artefacts from Downsample
    y1 = Cuts(y1, EstY, true);

    Nott::Data::Check::Size(x1, ("X"+std::to_string(tile) + " After cuts"));
    Nott::Data::Check::Size(y1, ("Y"+std::to_string(tile) + " After cuts"));
    x1 = Nott::Data::Transform::Format::Downsample(x1, {.size = {128, 128}, .interp = Nott::Data::Transform::Format::Options::InterpMode::Bilinear});
    y1 = Nott::Data::Transform::Format::Downsample(y1, {.size = {128, 128}, .interp = Nott::Data::Transform::Format::Options::InterpMode::Nearest});

    xs.push_back(x1);
    ys.push_back(y1);
    std::cout << "\n" <<std::endl;
  } std::cout << "\n" <<std::endl;
  torch::Tensor X = torch::cat(xs, 0);
  torch::Tensor Y = torch::cat(ys, 0);
  Nott::Data::Check::Size(X, "X Total");
  Nott::Data::Check::Size(Y, "Y Total");


  model.add(Nott::Layer::Conv2d({3, 32, {1, 3}, {1, 1}, {0, 1}},
                                Nott::Activation::ReLU,Nott::Initialization::HeUniform),"conv1");

  model.add(Nott::Layer::BatchNorm2d({.num_features = 32}), "bn1");
  model.add(Nott::Layer::Conv2d({32, 64, {1, 3}, {1, 1}, {0, 1}},
                                Nott::Activation::ReLU,Nott::Initialization::HeUniform),"conv2");

  model.add(Nott::Layer::BatchNorm2d({.num_features = 64}), "bn2");
  model.add(Nott::Layer::AdaptiveAvgPool2d({.output_size = {1,1}}), "gap");
  model.add(Nott::Layer::Flatten(), "flatten");
  model.add(Nott::Layer::FC({64, 6}, Nott::Activation::Identity, Nott::Initialization::XavierUniform), "classifier");

  model.links({
    Nott::LinkSpec{Nott::Port::Input("@input"), Nott::Port::Module("conv1")},
    Nott::LinkSpec{Nott::Port::Module("conv1"), Nott::Port::Module("bn1")},
    Nott::LinkSpec{Nott::Port::Module("bn1"), Nott::Port::Module("conv2")},
    Nott::LinkSpec{Nott::Port::Module("conv2"), Nott::Port::Module("bn2")},
    Nott::LinkSpec{Nott::Port::Module("bn2"), Nott::Port::Module("gap")},
    Nott::LinkSpec{Nott::Port::Module("gap"), Nott::Port::Module("flatten")},
    Nott::LinkSpec{Nott::Port::Module("flatten"), Nott::Port::Module("classifier")},
    Nott::LinkSpec{Nott::Port::Module("classifier"), Nott::Port::Output("@output")},
  }, true);




  Y = ConvertRgbMasksToOneHot(Y);
  X= X.to(torch::kFloat32) / 255.0f;

  //Nott::Plot::Data::Image(X, {0, 100, 1000, 2000, 5000, 11734});

  const auto total_training_samples = X.size(0);
  const auto B = 32;
  const auto E = 25;
  const auto steps_per_epoch = static_cast<std::size_t>((total_training_samples + B - 1) / B);
  const auto total_training_steps = std::max<std::size_t>(1, E * std::max<std::size_t>(steps_per_epoch, 1));


  model.set_loss(Nott::Loss::CrossEntropy({ /*.weight = w*/ }));
  model.set_optimizer(
    Nott::Optimizer::AdamW({.learning_rate = 5e-5, .weight_decay = 1e-4}),
    Nott::LrScheduler::CosineAnnealing({
      .T_max = (total_training_steps),
      .eta_min = 1e-6,
      .warmup_steps = std::min<std::size_t>(steps_per_epoch * 5, total_training_steps / 5),
      .warmup_start_factor = 0.1,
    }));

  model.set_regularization({Nott::Regularization::SWAG({
    .coefficient = 5e-4,
    .variance_epsilon = 1e-6,
    .start_step = static_cast<std::size_t>(0.65 * static_cast<double>(total_training_steps)),
    .accumulation_stride = std::max<std::size_t>(1, steps_per_epoch),
    .max_snapshots = 20,
  })});

  model.use_cuda(torch::cuda::is_available());

  // auto [Xt, Yt] = Nott::Data::Manipulation::Fraction(X, Y, 0.1f); // Test
  // std::tie(X, Y) = Nott::Data::Manipulation::Shuffle(X, Y);
  //
  // std::tie(X, Y) = Nott::Data::Manipulation::Fraction(X, Y, 0.1f); // 10%
  // std::tie(X, Y) = Nott::Data::Manipulation::CLAHE(X, Y, {256, 2.f, {4,4}, 1.f, true});

  SuperpixelParams params = {};
  params.desired_superpixels = 400;
  SuperpixelDataset ds = BuildSuperpixelDatasetForLimitedVoronoiCuts(X, Y, params, 20); //BuildSuperpixelDataset(X, Y, params);
  X = ds.X;
  Y = ds.y;
  Nott::Data::Check::Size(X, "X Train");
  X = X.view({-1, 3, 1, 5}).contiguous();
  Nott::Data::Check::Size(X, "X Train-ready");

  model.train(X, Y,
              {.epoch = E,
              .batch_size = B,
              .restore_best_state = true,
              //.test = std::vector<at::Tensor>{Xt, Yt},
              .graph_mode = Nott::GraphMode::Capture,
              .enable_amp = true,
              // .memory_format = torch::MemoryFormat::ChannelsLast
              });

  model.evaluate(X, Y, Nott::Evaluation::Classification,{
    Nott::Metric::Classification::Accuracy,
    Nott::Metric::Classification::Precision,
    Nott::Metric::Classification::Recall,
    Nott::Metric::Classification::F1,
    Nott::Metric::Classification::JaccardIndexMicro,
  },{.batch_size = B, .buffer_vram=2});

  // xs/ys store batches shaped [N, 3, H, W]; pick a single sample to visualize
  torch::Tensor sample_input = xs.front().index({0}).to(torch::kFloat32) / 255.0f;
  torch::Tensor sample_target = ys.front().index({0});

  cv::Mat sample_bgr = TensorCHWToBGRMat(sample_input);
  SuperpixelSegmentationResult sample_seg = RunSuperpixelSegmentation(sample_bgr, params);

  cv::Mat overlay = DrawVoronoiOverlay(sample_bgr, sample_seg.labels);
  torch::Tensor overlay_chw = BGRMatToTensorCHW(overlay);
  Nott::Plot::Data::Image(overlay_chw, {0});

  torch::Tensor sample_feats = ComputeSuperpixelRGBStats(
    sample_input,
    sample_seg.labels,
    static_cast<int>(sample_seg.centers.size()));
  sample_feats = sample_feats.view({-1, 3, 1, 5}).to(torch::kFloat32);
  if (torch::cuda::is_available()) {
    sample_feats = sample_feats.cuda();
  }

  torch::NoGradGuard no_grad;
  torch::Tensor sample_logits = model.forward(sample_feats);
  torch::Tensor sample_pred = sample_logits.argmax(1).to(torch::kCPU);

  torch::Tensor pred_map = torch::zeros(
    {sample_seg.labels.rows, sample_seg.labels.cols},
    torch::TensorOptions().dtype(torch::kLong));
  auto pred_acc = sample_pred.accessor<int64_t, 1>();
  auto map_acc = pred_map.accessor<int64_t, 2>();
  for (int y = 0; y < sample_seg.labels.rows; ++y) {
    const int* lbl_ptr = sample_seg.labels.ptr<int>(y);
    for (int x = 0; x < sample_seg.labels.cols; ++x) {
      const int sid = lbl_ptr[x];
      if (sid >= 0 && sid < sample_pred.size(0)) {
        map_acc[y][x] = pred_acc[sid];
      }
    }
  }


  auto forecast_rgb = ColorizeClassMap(pred_map);
  auto gt_one_hot = ConvertRgbMasksToOneHot(sample_target.unsqueeze(0));
  auto gt_rgb = ColorizeClassMap(gt_one_hot.argmax(1).squeeze(0));

  Nott::Plot::Data::Image(forecast_rgb, {0});
  Nott::Plot::Data::Image(gt_rgb, {0});
  Nott::Plot::Data::Image(sample_input, {0});
  // cv::Mat f_img(H, W, CV_8UC3, forecast_rgb.data_ptr<uint8_t>());
  // cv::Mat f_bgr;
  // cv::cvtColor(f_img, f_bgr, cv::COLOR_RGB2BGR);
  // cv::imwrite("/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile 1/forecast_raw.png", f_bgr);
  // cv::GaussianBlur(f_bgr, f_bgr, cv::Size(), 0.83);
  // cv::imwrite("/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile 1/forecast_smooth.png", f_bgr);

  return 0;
}


/*
 TODO: Use Hyperpixel from new cuts over Tiles
    Superpixels over cuts to move from H*W*3(Input) and H*W*6(output) to 5*3(input metrics) and 6 (argmax label output)
    compute cuts superpixel over B, and then in async Smoothing fractal to recreate Cuts labeling from Superpixel to Cuts and then from Cuts to Tile


    Target Skeleton:
    1) Async Cuts from Tiles
    2) Async Superpixels from Cuts
    3) Batchs Superpixels through Net via Metrics (Max, Min, Avg, Std, Skew; for R,G,B)
    4) Async Reconstruction of Cuts from Superpixels + Smoother
    5) Async Reconstruction of Tiles from Cuts + Smoother



    Size of the network will be so small that we will be able to compute mutliple Cuts at once.
    But low size Cuts help for K definition in Voronoi and Argmax over Superpixel labels

    Input&Output Improvement:
        Dimensions:
            Before I: 128*128*3 = 84,672  ||  O: 128*128*6 = 169,344
            After I: 5*3 = 15  ||  O: 6
            IMPROVEMENT I: 14111x  ||  O: 28223x
        Accuracy:
            Can't quantify, since high quantity of data is required (i don't have) + highly unbalanced class and TPR&TNR are treated egaly in loss, network learn how to avoid low-freq(complex) labels by never predicting them


    Reason:
        Tile reconstruction:
            Default Cuts:
                - More base-samples [Acc]
                - Lower Amount of Details [Acc]
                - Maintaining the Quality of Information ([(x/T)&(y/T)] -> bilinear) [Acc]
                - Lower Resolution [Latency]

            SubCuts:
                - Maximization of Base-Samples [Acc]
                - Redundant information can only appear twice in the dataset, but never at the same position on the image (always TopLeft vs BottomRight || TopRight vs BottomLeft)  leading to P(Overfit)⁻ + Conv=Sum[Freq(window)]=> taking neighbors and not all neurons to all neurons [Acc]

        Geodesic-Based Voronoi Diagram:
            - Able to use metrics as input (no Pics) [Lat+Acc]
                -> 14111 times less input neurons
            - Geodesic lead to P(Unique Class)^+ per superpixel, to use argmax(superpixel) leading to 6 output neurones [Lat+Acc]
                -> 28223 times less output neurons
            - Easy fix of argmax via smoothing during Cuts recreation (Async Post Process) [Acc]



        In Production:
            - Easy setup of asyncs for data Pre/Post-process
            - Superpixel computed by batches (With Hardware, even able to compute Superpixels from multiples Cuts inside a unique Batch)
            - Lightning fast Infra
            - Readable Infra (Data Cutting & Geodesic-Voronoi)
            - Voronoi make pre-segmentation of labels without Learning nor Data-Leakage




    Geodesic Distance Smoother:
        - RPCA
        - FFT + PSD
        - FFT_coef(0.05)
        - Marcenko Pastur w/ Eigen filter ?
    NB: Compute it over tile directly to make it faster
 */



/* // Test is same as train for Nabla bulbe
Epoch [1/50] | Train loss: 0.217209 | Test loss: 0.165778 | ΔLoss: N/A (∇) | duration: 95.63sec
Epoch [2/50] | Train loss: 0.163068 | Test loss: 0.154031 | ΔLoss: -0.011747 (∇) | duration: 95.79sec
Epoch [3/50] | Train loss: 0.152796 | Test loss: 0.149430 | ΔLoss: -0.004602 (∇) | duration: 95.11sec
Epoch [4/50] | Train loss: 0.146659 | Test loss: 0.145577 | ΔLoss: -0.003852 (∇) | duration: 94.93sec
Epoch [5/50] | Train loss: 0.140482 | Test loss: 0.137337 | ΔLoss: -0.008241 (∇) | duration: 95.01sec
Epoch [6/50] | Train loss: 0.130418 | Test loss: 0.124928 | ΔLoss: -0.012409 (∇) | duration: 93.56sec
Epoch [7/50] | Train loss: 0.122049 | Test loss: 0.116866 | ΔLoss: -0.008062 (∇) | duration: 93.50sec
Epoch [8/50] | Train loss: 0.115397 | Test loss: 0.105388 | ΔLoss: -0.011478 (∇) | duration: 93.50sec
Epoch [9/50] | Train loss: 0.114268 | Test loss: 0.127917 | ΔLoss: +0.022529 (∇) | duration: 93.93sec
Epoch [10/50] | Train loss: 0.110198 | Test loss: 0.095986 | ΔLoss: -0.009402 (∇) | duration: 105.43sec
Epoch [11/50] | Train loss: 0.101153 | Test loss: 0.103741 | ΔLoss: +0.007755 (∇) | duration: 106.69sec
Epoch [12/50] | Train loss: 0.092748 | Test loss: 0.086289 | ΔLoss: -0.009697 (∇) | duration: 106.17sec
Epoch [13/50] | Train loss: 0.088982 | Test loss: 0.086340 | ΔLoss: +0.000051 (∇) | duration: 103.05sec
Epoch [14/50] | Train loss: 0.081105 | Test loss: 0.073103 | ΔLoss: -0.013187 (∇) | duration: 103.34sec
Epoch [15/50] | Train loss: 0.078934 | Test loss: 0.072175 | ΔLoss: -0.000927 (∇) | duration: 104.06sec
Epoch [16/50] | Train loss: 0.075805 | Test loss: 0.091221 | ΔLoss: +0.019045 (∇) | duration: 96.86sec
Epoch [17/50] | Train loss: 0.070581 | Test loss: 0.065021 | ΔLoss: -0.007154 (∇) | duration: 100.52sec
Epoch [18/50] | Train loss: 0.072420 | Test loss: 0.064317 | ΔLoss: -0.000704 (∇) | duration: 95.38sec
Epoch [19/50] | Train loss: 0.071357 | Test loss: 0.065981 | ΔLoss: +0.001664 (∇) | duration: 98.08sec
Epoch [20/50] | Train loss: 0.063595 | Test loss: 0.055153 | ΔLoss: -0.009164 (∇) | duration: 94.51sec
Epoch [21/50] | Train loss: 0.053435 | Test loss: 0.048825 | ΔLoss: -0.006328 (∇) | duration: 96.21sec
Epoch [22/50] | Train loss: 0.049913 | Test loss: 0.046706 | ΔLoss: -0.002120 (∇) | duration: 93.60sec
Epoch [23/50] | Train loss: 0.047069 | Test loss: 0.042982 | ΔLoss: -0.003723 (∇) | duration: 93.59sec
Epoch [24/50] | Train loss: 0.042955 | Test loss: 0.040060 | ΔLoss: -0.002923 (∇) | duration: 94.58sec
Epoch [25/50] | Train loss: 0.040667 | Test loss: 0.038410 | ΔLoss: -0.001649 (∇) | duration: 95.96sec
Epoch [26/50] | Train loss: 0.037267 | Test loss: 0.034624 | ΔLoss: -0.003786 (∇) | duration: 98.51sec
Epoch [27/50] | Train loss: 0.037847 | Test loss: 0.033765 | ΔLoss: -0.000859 (∇) | duration: 98.19sec
Epoch [28/50] | Train loss: 0.033383 | Test loss: 0.029803 | ΔLoss: -0.003962 (∇) | duration: 93.50sec
Epoch [29/50] | Train loss: 0.030588 | Test loss: 0.027783 | ΔLoss: -0.002021 (∇) | duration: 94.13sec
Epoch [30/50] | Train loss: 0.029718 | Test loss: 0.027049 | ΔLoss: -0.000734 (∇) | duration: 93.50sec
Epoch [31/50] | Train loss: 0.026721 | Test loss: 0.024883 | ΔLoss: -0.002165 (∇) | duration: 93.81sec
Epoch [32/50] | Train loss: 0.024736 | Test loss: 0.022889 | ΔLoss: -0.001994 (∇) | duration: 93.51sec
Epoch [33/50] | Train loss: 0.023193 | Test loss: 0.021997 | ΔLoss: -0.000892 (∇) | duration: 93.47sec
Epoch [34/50] | Train loss: 0.021912 | Test loss: 0.022896 | ΔLoss: +0.000898 (∇) | duration: 96.04sec
Epoch [35/50] | Train loss: 0.021031 | Test loss: 0.019439 | ΔLoss: -0.002558 (∇) | duration: 96.07sec
Epoch [36/50] | Train loss: 0.019589 | Test loss: 0.019446 | ΔLoss: +0.000007 (∇) | duration: 96.08sec
Epoch [37/50] | Train loss: 0.018850 | Test loss: 0.018018 | ΔLoss: -0.001421 (∇) | duration: 96.09sec
Epoch [38/50] | Train loss: 0.017995 | Test loss: 0.016745 | ΔLoss: -0.001272 (∇) | duration: 96.09sec
Epoch [39/50] | Train loss: 0.017139 | Test loss: 0.016592 | ΔLoss: -0.000154 (∇) | duration: 96.09sec
Epoch [40/50] | Train loss: 0.016656 | Test loss: 0.015644 | ΔLoss: -0.000948 (∇) | duration: 97.35sec
Epoch [41/50] | Train loss: 0.016203 | Test loss: 0.015593 | ΔLoss: -0.000050 (∇) | duration: 97.24sec
Epoch [42/50] | Train loss: 0.015494 | Test loss: 0.014841 | ΔLoss: -0.000752 (∇) | duration: 97.25sec
Epoch [43/50] | Train loss: 0.014906 | Test loss: 0.014285 | ΔLoss: -0.000557 (∇) | duration: 97.66sec
Epoch [44/50] | Train loss: 0.014457 | Test loss: 0.013886 | ΔLoss: -0.000399 (∇) | duration: 97.23sec
Epoch [45/50] | Train loss: 0.014038 | Test loss: 0.013522 | ΔLoss: -0.000364 (∇) | duration: 97.23sec
Epoch [46/50] | Train loss: 0.013667 | Test loss: 0.013494 | ΔLoss: -0.000028 (∇) | duration: 97.39sec
Epoch [47/50] | Train loss: 0.013382 | Test loss: 0.012975 | ΔLoss: -0.000519 (∇) | duration: 96.60sec
Epoch [48/50] | Train loss: 0.013139 | Test loss: 0.012817 | ΔLoss: -0.000158 (∇) | duration: 96.08sec
Epoch [49/50] | Train loss: 0.012878 | Test loss: 0.012620 | ΔLoss: -0.000197 (∇) | duration: 96.08sec
Epoch [50/50] | Train loss: 0.012654 | Test loss: 0.012409 | ΔLoss: -0.000211 (∇) | duration: 97.82sec
[Nott] Reloading best state of the network...
End
Test Inputs: (11735, 3, 128, 128)
Test Targets: (11735, 128, 128)

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Evaluation: Classification ┃          ┃                    ┃
┣━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━┫
┃ Metric                ┃    Macro ┃ Weighted (support) ┃
┣━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━┫
┃ Accuracy              ┃ 0.982847 ┃           0.970561 ┃
┃ Precision             ┃ 0.895484 ┃           0.953850 ┃
┃ Recall                ┃ 0.971272 ┃           0.948542 ┃
┃ Jaccard index (micro) ┃ 0.902121 ┃           0.902121 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━┛

Overall accuracy (micro): 0.948542

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Per-class             ┃          ┃          ┃           ┃          ┃          ┃          ┃
┣━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Metric                ┃  Label 0 ┃  Label 1 ┃   Label 2 ┃  Label 3 ┃  Label 4 ┃  Label 5 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Support               ┃ 26572179 ┃ 17905191 ┃ 100457165 ┃   927471 ┃ 26449061 ┃ 19955173 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Accuracy              ┃ 0.984740 ┃ 0.974361 ┃  0.955777 ┃ 0.998528 ┃ 0.996012 ┃ 0.987667 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Precision             ┃ 0.922010 ┃ 0.798134 ┃  0.992994 ┃ 0.766406 ┃ 0.978783 ┃ 0.914579 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Recall                ┃ 0.971787 ┃ 0.970029 ┃  0.921864 ┃ 0.999481 ┃ 0.992522 ┃ 0.971949 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Jaccard index (micro) ┃ 0.897973 ┃ 0.778926 ┃  0.915907 ┃ 0.766102 ┃ 0.971617 ┃ 0.891059 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┛




// Only 6 epochs over 50 led to non-improvement
*/




