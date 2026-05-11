#ifndef Nott_PLOT_DETAILS_DATA_HPP
#define Nott_PLOT_DETAILS_DATA_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <atomic>
#include <stdexcept>
#include <string>
#include <functional>
#include <limits>
#include <utility>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cctype>
#include <iterator>

#include <torch/torch.h>

#include "../../utils/gnuplot.hpp"

namespace Nott::Plot::Data {
    namespace Details {
        inline auto as_cpu_contiguous(const torch::Tensor& tensor) -> torch::Tensor
        {
            if (!tensor.defined()) {
                throw std::invalid_argument("Plot::Data tensor must be defined");
            }
            auto result = tensor;
            if (result.device().is_cuda()) {
                result = result.cpu();
            }
            if (!result.is_contiguous()) {
                result = result.contiguous();
            }
            return result;
        }

        inline auto flatten_to_double_vector(const torch::Tensor& tensor) -> std::vector<double>
        {
            auto contiguous = as_cpu_contiguous(tensor);
            contiguous = contiguous.to(torch::kFloat32);
            std::vector<double> values;
            values.reserve(static_cast<std::size_t>(contiguous.numel()));
            auto flat = contiguous.view({contiguous.numel()});
            auto accessor = flat.accessor<float, 1>();

            for (int64_t i = 0; i < contiguous.numel(); ++i) {
                values.push_back(static_cast<double>(accessor[i]));
            }
            return values;
        }

        inline auto build_color_palette() -> std::vector<std::string>
        {
            static const std::array<const char*, 10> palette = {
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"};
            return {palette.begin(), palette.end()};
        }
        inline auto compute_series_bounds(const std::vector<std::vector<double>>& series)
            -> std::pair<double, double>
        {
            if (series.empty()) {
                return {0.0, 0.0};
            }
            double minValue = std::numeric_limits<double>::infinity();
            double maxValue = -std::numeric_limits<double>::infinity();
            for (const auto& values : series) {
                for (double value : values) {
                    minValue = std::min(minValue, value);
                    maxValue = std::max(maxValue, value);
                }
            }

            if (!std::isfinite(minValue) || !std::isfinite(maxValue)) {
                return {0.0, 0.0};
            }
            if (minValue == maxValue) {
                constexpr double epsilon = 1.0;
                return {minValue - epsilon, maxValue + epsilon};
            }
            return {minValue, maxValue};
        }

        inline auto escape_single_quotes(const std::string& input) -> std::string
        {
            std::string escaped;
            escaped.reserve(input.size());
            for (char ch : input) {
                if (ch == static_cast<char>(39)) {
                    escaped += "\\'";
                } else {
                    escaped += ch;
                }
            }
            return escaped;
        }

        inline auto prepare_grayscale_tensor(torch::Tensor tensor) -> torch::Tensor
        {
            tensor = as_cpu_contiguous(std::move(tensor));
            tensor = tensor.to(torch::kFloat32);
            if (tensor.dim() == 3) {
                if (tensor.size(0) == 1) {
                    tensor = tensor.squeeze(0);
                } else if (tensor.size(2) == 1) {
                    tensor = tensor.squeeze(2);
                }
            }
            if (tensor.dim() != 2) {
                throw std::invalid_argument("Image expects grayscale tensors to be 2D after squeezing");
            }
            return tensor.contiguous();
        }

        inline auto prepare_color_tensor(torch::Tensor tensor) -> torch::Tensor
        {
            tensor = as_cpu_contiguous(std::move(tensor));
            tensor = tensor.to(torch::kFloat32);
            if (tensor.dim() == 3) {
                if (tensor.size(0) == 3 || tensor.size(0) == 4) {
                    if (tensor.size(0) == 4) {
                        tensor = tensor.index({torch::indexing::Slice(0, 3), torch::indexing::Ellipsis});
                    }
                } else if (tensor.size(2) == 3 || tensor.size(2) == 4) {
                    tensor = tensor.permute({2, 0, 1}).contiguous();
                    if (tensor.size(0) == 4) {
                        tensor = tensor.index({torch::indexing::Slice(0, 3), torch::indexing::Ellipsis});
                    }
                } else {
                    throw std::invalid_argument("Unsupported color tensor layout");
                }
            } else {
                throw std::invalid_argument("Image expects color tensors to be 3D");
            }
            return tensor.contiguous();
        }

        inline auto compute_rgb_scaling(const torch::Tensor& tensor) -> std::pair<double, double>
        {
            auto minTensor = tensor.min();
            auto maxTensor = tensor.max();
            const double minValue = minTensor.item<double>();
            const double maxValue = maxTensor.item<double>();
            double offset = 0.0;
            double scale = 1.0;

            if (maxValue <= 1.0 && minValue >= 0.0) {
                scale = 255.0;
            } else if (maxValue > 255.0 || minValue < 0.0) {
                offset = -minValue;
                const double range = maxValue + offset;
                if (range > 0.0) {
                    scale = 255.0 / range;
                } else {
                    scale = 255.0;
                }
            }
            return {scale, offset};
        }

        inline auto build_grayscale_writer(const torch::Tensor& tensor)
            -> std::function<void(std::FILE*)>
        {
            auto prepared = prepare_grayscale_tensor(tensor.clone());
            const auto height = prepared.size(0);
            const auto width = prepared.size(1);
            return [prepared, height, width](std::FILE* pipe) {
                auto accessor = prepared.accessor<float, 2>();
                for (int64_t row = 0; row < height; ++row) {
                    for (int64_t col = 0; col < width; ++col) {
                        if (std::fprintf(pipe,
                                          "%lld %lld %.*g\n",
                                          static_cast<long long>(col),
                                          static_cast<long long>(row),
                                          15,
                                          static_cast<double>(accessor[row][col])) < 0) {
                            throw std::runtime_error("Failed to write grayscale image data to gnuplot");
                                          }
                    }
                    if (std::fprintf(pipe, "\n") < 0) {
                        throw std::runtime_error("Failed to write grayscale row separator to gnuplot");
                    }
                }
            };
        }

        inline auto build_color_writer(const torch::Tensor& tensor)
            -> std::function<void(std::FILE*)>
        {
            auto prepared = prepare_color_tensor(tensor.clone());
            const auto height = prepared.size(1);
            const auto width = prepared.size(2);
            const auto [scale, offset] = compute_rgb_scaling(prepared);
            return [prepared, height, width, scale, offset](std::FILE* pipe) {
                auto accessor = prepared.accessor<float, 3>();
                for (int64_t row = 0; row < height; ++row) {
                    for (int64_t col = 0; col < width; ++col) {
                        const auto r = std::clamp(
                            std::lround((static_cast<double>(accessor[0][row][col]) + offset) * scale),
                            0L,
                            255L);
                        const auto g = std::clamp(
                            std::lround((static_cast<double>(accessor[1][row][col]) + offset) * scale),
                            0L,
                            255L);
                        const auto b = std::clamp(
                            std::lround((static_cast<double>(accessor[2][row][col]) + offset) * scale),
                            0L,
                            255L);
                        if (std::fprintf(pipe,
                                          "%lld %lld %ld %ld %ld\n",
                                          static_cast<long long>(col),
                                          static_cast<long long>(row),
                                          r,
                                          g,
                                          b) < 0) {
                            throw std::runtime_error("Failed to write color image data to gnuplot");
                                          }
                    }
                    if (std::fprintf(pipe, "\n") < 0) {
                        throw std::runtime_error("Failed to write color row separator to gnuplot");
                    }
                }
            };
        }

        inline auto format_double(double value) -> std::string
        {
            std::ostringstream stream;
            stream << std::setprecision(10) << value;
            return stream.str();
        }
    }

    struct TimeseriePlotOptions {
        std::string title{"Timeseries"};
        std::string xLabel{"Time"};
        std::string yLabel{"Value"};
        std::vector<std::string> seriesTitles{};
        std::optional<std::string> testSeparatorLabel{std::string("Test separator")};
        bool showGrid{true};
    };

    class Timeserie;
    void Render(const Timeserie& timeserie, const TimeseriePlotOptions& options = {});

    class Timeserie {
    private:
        std::vector<double> m_xAxis{};
        std::vector<std::vector<double>> m_series{};
        std::vector<std::string> m_colors{};
        std::optional<double> m_testSeparator{};

    public:
        Timeserie(torch::Tensor xtrain,
                  std::optional<torch::Tensor> xtest = std::nullopt,
                  std::optional<TimeseriePlotOptions> renderOptions = std::nullopt)
        {
            initialize(std::move(xtrain), std::move(xtest));
            if (renderOptions.has_value())
                Render(*this, *renderOptions);
            else
                Render(*this, TimeseriePlotOptions{});
        }

        [[nodiscard]] auto xAxis() const noexcept -> const std::vector<double>& { return m_xAxis; }
        [[nodiscard]] auto series() const noexcept -> const std::vector<std::vector<double>>& { return m_series; }
        [[nodiscard]] auto colors() const noexcept -> const std::vector<std::string>& { return m_colors; }
        [[nodiscard]] auto testSeparator() const noexcept -> const std::optional<double>& { return m_testSeparator; }
        [[nodiscard]] auto hasMultipleInputs() const noexcept -> bool { return m_series.size() > 1; }

    private:
        void initialize(torch::Tensor xtrain, std::optional<torch::Tensor> xtest)
        {
            if (!xtrain.defined()) throw std::invalid_argument("Timeserie xtrain must be defined");
            xtrain = Details::as_cpu_contiguous(xtrain).to(torch::kFloat32);

            /// Concatenate if xtest exists
            std::size_t train_len = static_cast<std::size_t>(xtrain.size(0));
            if (xtest && xtest->defined()) {
                auto test_cpu = Details::as_cpu_contiguous(*xtest).to(torch::kFloat32);
                if (xtrain.dim() != test_cpu.dim())
                    throw std::invalid_argument("xtrain and xtest must have same dimensionality");
                xtrain = torch::cat({xtrain, test_cpu}, 0);
                m_testSeparator = static_cast<double>(train_len);
            }

            /// Extract series (1D or 2D)
            if (xtrain.dim() == 1) {
                auto acc = xtrain.accessor<float, 1>();
                m_series.resize(1);
                m_series[0].reserve(xtrain.size(0));
                for (int64_t i = 0; i < xtrain.size(0); ++i)
                    m_series[0].push_back(static_cast<double>(acc[i]));
            } else if (xtrain.dim() == 2) {
                const auto length = static_cast<std::size_t>(xtrain.size(0));
                const auto features = static_cast<std::size_t>(xtrain.size(1));
                auto acc = xtrain.accessor<float, 2>();
                m_series.resize(features);
                for (std::size_t f = 0; f < features; ++f) {
                    m_series[f].reserve(length);
                    for (int64_t t = 0; t < xtrain.size(0); ++t)
                        m_series[f].push_back(static_cast<double>(acc[t][static_cast<int64_t>(f)]));
                }
            } else {
                throw std::invalid_argument("Timeserie expects 1D or 2D tensor");
            }

            /// Build X-axis
            const auto total_len = m_series.empty() ? 0 : m_series.front().size();
            m_xAxis.resize(total_len);
            for (std::size_t i = 0; i < total_len; ++i)
                m_xAxis[i] = static_cast<double>(i);

            /// Assign colors
            auto palette = Details::build_color_palette();
            m_colors.reserve(m_series.size());
            for (std::size_t idx = 0; idx < m_series.size(); ++idx)
                m_colors.push_back(palette[idx % palette.size()]);
        }
    };

    inline void Render(const Timeserie& timeserie, const TimeseriePlotOptions& options)
    {
        Utils::Gnuplot plotter{};
        plotter.setMouse(true);
        if (!options.title.empty()) plotter.setTitle(options.title);
        if (!options.xLabel.empty()) plotter.setXLabel(options.xLabel);
        if (!options.yLabel.empty()) plotter.setYLabel(options.yLabel);
        if (options.showGrid) plotter.setGrid(true);

        const auto& xAxis = timeserie.xAxis();
        const auto& series = timeserie.series();
        const auto& colors = timeserie.colors();

        std::vector<Utils::Gnuplot::DataSet2D> datasets;
        datasets.reserve(series.size() + (timeserie.testSeparator().has_value() ? 1 : 0));

        for (std::size_t idx = 0; idx < series.size(); ++idx) {
            Utils::Gnuplot::DataSet2D d{};
            d.x = xAxis;
            d.y = series[idx];
            d.title = (idx < options.seriesTitles.size() && !options.seriesTitles[idx].empty())
                        ? options.seriesTitles[idx]
                        : "Series " + std::to_string(idx + 1);
            Utils::Gnuplot::PlotStyle s{};
            s.mode = Utils::Gnuplot::PlotMode::Lines;
            s.lineColor = colors[idx % colors.size()];
            s.lineWidth = 2.0;
            d.style = std::move(s);
            datasets.push_back(std::move(d));
        }

        if (timeserie.testSeparator()) {
            const auto [minv, maxv] = Details::compute_series_bounds(series);
            Utils::Gnuplot::DataSet2D sep{};
            const double sepX = *timeserie.testSeparator();
            sep.x = {sepX, sepX};
            sep.y = {minv, maxv};
            sep.title = options.testSeparatorLabel.value_or("Test split");
            Utils::Gnuplot::PlotStyle s{};
            s.mode = Utils::Gnuplot::PlotMode::Lines;
            s.lineColor = "#000000";
            s.lineWidth = 1.5;
            s.extra = "dashtype 2";
            sep.style = std::move(s);
            datasets.push_back(std::move(sep));
        }

        if (datasets.empty())
            throw std::runtime_error("Timeserie::Render requires at least one dataset");

        plotter.plot(datasets);
    }

struct MatrixOptions {
        bool digits{false};
        std::string color{"binary"};
    };

    inline void Matrix(torch::Tensor matrix, std::optional<MatrixOptions> plotOptions = std::nullopt)
    {
        MatrixOptions options = plotOptions.value_or(MatrixOptions{});

        if (!matrix.defined()) {
            throw std::invalid_argument("Matrix tensor must be defined");
        }

        matrix = Details::as_cpu_contiguous(std::move(matrix)).to(torch::kFloat32);
        if (matrix.dim() == 1) {
            matrix = matrix.unsqueeze(0);
        }
        if (matrix.dim() != 2) {
            throw std::invalid_argument("Matrix expects a 2D tensor");
        }

        const auto height = static_cast<std::size_t>(matrix.size(0));
        const auto width = static_cast<std::size_t>(matrix.size(1));
        if (height == 0 || width == 0) {
            throw std::invalid_argument("Matrix expects non-empty dimensions");
        }

        auto prepared = matrix.contiguous();

        const double minValue = prepared.min().item<double>();
        const double maxValue = prepared.max().item<double>();

        Utils::Gnuplot plotter{};
        plotter.setMouse(false);
        plotter.command("unset key");
        plotter.command("set view map");
        plotter.command("set tics scale 0");
        plotter.command("set colorbox");

        auto formatRange = [](double value) {
            return Details::format_double(value);
        };

        {
            std::ostringstream xrange;
            xrange << "set xrange [" << formatRange(-0.5) << ':'
                   << formatRange(static_cast<double>(width) - 0.5) << ']';
            plotter.command(xrange.str());
        }

        {
            std::ostringstream yrange;
            yrange << "set yrange ["
                   << formatRange(static_cast<double>(height) - 0.5) << ':'
                   << formatRange(-0.5) << ']';
            plotter.command(yrange.str());
        }

        {
            std::ostringstream xtics;
            xtics << "set xtics 0,1," << (width > 0 ? static_cast<long long>(width - 1) : 0);
            plotter.command(xtics.str());
        }

        {
            std::ostringstream ytics;
            ytics << "set ytics 0,1," << (height > 0 ? static_cast<long long>(height - 1) : 0);
            plotter.command(ytics.str());
        }

        const auto paletteName = [&]() {
            std::string lowered;
            lowered.reserve(options.color.size());
            std::transform(options.color.begin(), options.color.end(), std::back_inserter(lowered), [](unsigned char ch) {
                return static_cast<char>(std::tolower(ch));
            });
            if (lowered == "binary") {
                return std::string{"defined (0 '#ffffff', 1 '#000000')"};
            }
            if (lowered == "grey") {
                return std::string{"gray"};
            }
            return options.color;
        }();

        if (!paletteName.empty()) {
            plotter.setPalette(paletteName);
        } else {
            plotter.unsetPalette();
        }

        if (std::isfinite(minValue) && std::isfinite(maxValue)) {
            if (maxValue > minValue) {
                std::ostringstream cbrange;
                cbrange << "set cbrange [" << formatRange(minValue) << ':' << formatRange(maxValue) << ']';
                plotter.command(cbrange.str());
            } else {
                const double epsilon = std::max(1.0, std::abs(maxValue) * 0.1);
                std::ostringstream cbrange;
                cbrange << "set cbrange [" << formatRange(maxValue - epsilon) << ':'
                        << formatRange(maxValue + epsilon) << ']';
                plotter.command(cbrange.str());
            }
        } else {
            plotter.command("unset cbrange");
        }

        static std::atomic<std::size_t> matrixCounter{0};
        const auto datablockId = "Nott_matrix_" + std::to_string(++matrixCounter);
        const auto datablockRef = "$" + datablockId;
        plotter.defineDatablock(datablockId, Details::build_grayscale_writer(prepared));

        if (options.digits) {
            plotter.command("unset label");
            const double contrastThreshold = (minValue + maxValue) / 2.0;
            const bool useContrast = std::isfinite(contrastThreshold) && (maxValue > minValue);
            auto accessor = prepared.accessor<float, 2>();
            for (std::size_t row = 0; row < height; ++row) {
                for (std::size_t col = 0; col < width; ++col) {
                    const double value = static_cast<double>(accessor[row][col]);
                    std::ostringstream textStream;
                    textStream << std::setprecision(3) << value;
                    auto text = Details::escape_single_quotes(textStream.str());
                    std::ostringstream label;
                    label << "set label '" << text << "' at "
                          << formatRange(static_cast<double>(col)) << ','
                          << formatRange(static_cast<double>(row))
                          << " center front";
                    if (useContrast) {
                        const bool bright = value > contrastThreshold;
                        label << " tc rgb '" << (bright ? "#ffffff" : "#000000") << "'";
                    }
                    plotter.command(label.str());
                }
            }
        }

        std::ostringstream plotCommand;
        plotCommand << "plot " << datablockRef << " using 1:2:3 with image";
        plotter.command(plotCommand.str());
    }


    struct ImagePlotOptions {
        std::string layoutTitle{"Selected images"};
        std::vector<std::string> imageTitles{};
        bool showColorBox{false};
        std::optional<int> columns{};
        std::optional<int> rows{};
    };

    class Image;
    void Render(const Image& images, const ImagePlotOptions& options = {});

    class Image {
    private:
        std::vector<std::size_t> m_indices{};
        std::vector<torch::Tensor> m_selectedImages{};
        bool m_isColor{false};
    public:
        Image(torch::Tensor images, std::vector<std::size_t> indices, std::optional<ImagePlotOptions> renderOptions = std::nullopt) : m_indices(std::move(indices)) {
            initialize(std::move(images));
            if (renderOptions.has_value()) {
                Render(*this, *renderOptions);
            } else {
                Render(*this, ImagePlotOptions{});
            }
        }

        [[nodiscard]] auto selected() const noexcept -> const std::vector<torch::Tensor>& {
            return m_selectedImages;
        }

        [[nodiscard]] auto indices() const noexcept -> const std::vector<std::size_t>& {
            return m_indices;
        }

        [[nodiscard]] auto isColor() const noexcept -> bool {
            return m_isColor;
        }

    private:
        void initialize(torch::Tensor images)
        {
            if (!images.defined()) {
                throw std::invalid_argument("Image tensor must be defined");
            }
            images = Details::as_cpu_contiguous(images);

            if (images.dim() < 3 || images.dim() > 4) {
                throw std::invalid_argument("Image expects a 3D or 4D tensor");
            }
            if (images.dim() == 3) {
                const auto dim0 = static_cast<std::size_t>(images.size(0));
                const auto dim2 = static_cast<std::size_t>(images.size(2));
                const bool channelFirstColor = dim0 == 3 || dim0 == 4;
                const bool channelLastColor = dim2 == 3 || dim2 == 4;

                if (channelFirstColor || channelLastColor) {
                    m_isColor = true;
                    auto normalized = channelLastColor && !channelFirstColor
                                          ? images.permute({2, 0, 1}).contiguous()
                                          : images;
                    if (normalized.size(0) == 4) {
                        normalized = normalized.index({torch::indexing::Slice(0, 3), torch::indexing::Ellipsis});
                    }

                    const std::size_t batchSize = 1;
                    for (auto index : m_indices) {
                        if (index >= batchSize) {
                            throw std::out_of_range("Requested image index out of bounds");
                        }
                        (void)index;
                        m_selectedImages.push_back(normalized.detach().clone());
                    }
                    return;
                }
            }


            const auto batchSize = static_cast<std::size_t>(images.size(0));
            for (auto index : m_indices) {
                if (index >= batchSize) {
                    throw std::out_of_range("Requested image index out of bounds");
                }
            }

            if (images.dim() == 4) {
                const auto channels = static_cast<std::size_t>(images.size(1));
                const auto trailingChannels = static_cast<std::size_t>(images.size(3));
                m_isColor = channels > 1 || trailingChannels == 3 || trailingChannels == 4;
                for (auto index : m_indices) {
                    auto slice = images[index].detach().clone();
                    m_selectedImages.push_back(std::move(slice));
                }
            } else { // Grayscale path
                m_isColor = false;
                for (auto index : m_indices) {
                    auto slice = images[index].detach().clone();
                    m_selectedImages.push_back(std::move(slice));
                }
            }
        }


    };



    inline void Render(const Image& images, const ImagePlotOptions& options) {
        const auto& selected = images.selected();
        if (selected.empty()) {
            throw std::invalid_argument("Image::Render requires at least one image");
        }

        const std::size_t count = selected.size();
        int columns = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(count))));
        int rows = static_cast<int>((static_cast<int>(count) + columns - 1) / columns);
        if (options.columns) {
            columns = std::max(1, *options.columns);
            rows = static_cast<int>((static_cast<int>(count) + columns - 1) / columns);
        }
        if (options.rows) {
            rows = std::max(1, *options.rows);
        }

        Utils::Gnuplot plotter{};
        plotter.setMouse(false);
        std::ostringstream multiplotOptions;
        multiplotOptions << "layout " << rows << ',' << columns;
        if (!options.layoutTitle.empty()) {
            multiplotOptions << " title '" << Details::escape_single_quotes(options.layoutTitle) << "'";
        }
        plotter.beginMultiplot(multiplotOptions.str());
        plotter.command(options.showColorBox ? "set colorbox" : "unset colorbox");
        plotter.command("unset key");
        plotter.command("set view map");
        plotter.command("set size ratio -1");
        plotter.command("unset xtics");
        plotter.command("unset ytics");

        static std::atomic<std::size_t> datablockCounter{0}; // warning: Reading from '-' inside a multiplot not supported; use a datablock instead

        for (std::size_t idx = 0; idx < selected.size(); ++idx) {
            const auto& tensor = selected[idx];
            const auto imageIndex = images.indices()[idx];
            std::string title;
            if (idx < options.imageTitles.size() && !options.imageTitles[idx].empty()) {
                title = options.imageTitles[idx];
            } else {
                title = "Image " + std::to_string(imageIndex);
            }
            plotter.setTitle(title);

            if (images.isColor()) {
                auto prepared = Details::prepare_color_tensor(tensor.clone());
                const auto height = prepared.size(1);
                const auto width = prepared.size(2);

                const auto datablockId = "Nott_image_" + std::to_string(++datablockCounter);
                const auto datablockRef = "$" + datablockId;

                std::ostringstream xrange;
                xrange << "set xrange [0:" << (width - 1) << ']';
                plotter.command(xrange.str());
                std::ostringstream yrange;
                yrange << "set yrange [" << (height - 1) << ":0]";
                plotter.command(yrange.str());
                plotter.command("set size ratio -1");

                plotter.defineDatablock(datablockId, Details::build_color_writer(prepared));
                std::ostringstream plotCommand;
                plotCommand << "plot " << datablockRef << " using 1:2:3:4:5 with rgbimage";
                plotter.command(plotCommand.str());
            } else {
                auto prepared = Details::prepare_grayscale_tensor(tensor.clone());
                const auto height = prepared.size(0);
                const auto width = prepared.size(1);
                const auto datablockId = "Nott_image_" + std::to_string(++datablockCounter);
                const auto datablockRef = "$" + datablockId;

                std::ostringstream xrange;
                xrange << "set xrange [0:" << (width - 1) << ']';
                plotter.command(xrange.str());
                std::ostringstream yrange;
                yrange << "set yrange [" << (height - 1) << ":0]";
                plotter.command(yrange.str());
                plotter.command("set size ratio -1");

                const auto minValue = prepared.min().item<double>();
                const auto maxValue = prepared.max().item<double>();
                if (std::isfinite(minValue) && std::isfinite(maxValue) && maxValue > minValue) {
                    std::ostringstream cbrange;
                    cbrange << "set cbrange [" << Details::format_double(minValue) << ':'
                            << Details::format_double(maxValue) << ']';
                    plotter.command(cbrange.str());
                } else {
                    plotter.command("unset cbrange");
                }
                plotter.command("set palette gray");

                plotter.defineDatablock(datablockId, Details::build_grayscale_writer(prepared));
                std::ostringstream plotCommand;
                plotCommand << "plot " << datablockRef << " using 1:2:3 with image";
                plotter.command(plotCommand.str());
            }
        }

        plotter.endMultiplot();
    }
}

#endif // Nott_PLOT_DETAILS_DATA_HPP