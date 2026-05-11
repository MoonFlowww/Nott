#ifndef LIBNott_TERMINAL_HPP
#define LIBNott_TERMINAL_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace Nott::Utils::Terminal {
    /// ---------- Colors ----------
    namespace Colors {
        inline constexpr std::string_view kReset = "\033[0m";

        /// 8/16-color foregrounds
        inline constexpr std::string_view kBlack         = "\033[30m";
        inline constexpr std::string_view kRed           = "\033[31m";
        inline constexpr std::string_view kGreen         = "\033[32m";
        inline constexpr std::string_view kYellow        = "\033[33m";
        inline constexpr std::string_view kBlue          = "\033[34m";
        inline constexpr std::string_view kMagenta       = "\033[35m";
        inline constexpr std::string_view kCyan          = "\033[36m";
        inline constexpr std::string_view kWhite         = "\033[37m";

        inline constexpr std::string_view kBrightBlack   = "\033[90m";
        inline constexpr std::string_view kBrightRed     = "\033[91m";
        inline constexpr std::string_view kBrightGreen   = "\033[92m";
        inline constexpr std::string_view kBrightYellow  = "\033[93m";
        inline constexpr std::string_view kBrightBlue    = "\033[94m";
        inline constexpr std::string_view kBrightMagenta = "\033[95m";
        inline constexpr std::string_view kBrightCyan    = "\033[96m";
        inline constexpr std::string_view kBrightWhite   = "\033[97m";

        /// Named 256-color convenience
        inline constexpr std::string_view kTurquoise    = "\033[38;5;49m";
        inline constexpr std::string_view kCrimson      = "\033[38;5;196m";
        inline constexpr std::string_view kAzure        = "\033[38;5;33m";
        inline constexpr std::string_view kOrange       = "\033[38;5;208m";
        inline constexpr std::string_view kDimOrange    = "\033[2;38;5;208m";
        inline constexpr std::string_view kIvory        = "\033[38;5;15m";
        inline constexpr std::string_view kDimIvory     = "\033[2;38;5;15m";
        inline constexpr std::string_view kGoldenrod    = "\033[38;5;221m";
    }

    namespace BgColors {
        /// 8/16-color backgrounds
        inline constexpr std::string_view kBlack         = "\033[40m";
        inline constexpr std::string_view kRed           = "\033[41m";
        inline constexpr std::string_view kGreen         = "\033[42m";
        inline constexpr std::string_view kYellow        = "\033[43m";
        inline constexpr std::string_view kBlue          = "\033[44m";
        inline constexpr std::string_view kMagenta       = "\033[45m";
        inline constexpr std::string_view kCyan          = "\033[46m";
        inline constexpr std::string_view kWhite         = "\033[47m";

        inline constexpr std::string_view kBrightBlack   = "\033[100m";
        inline constexpr std::string_view kBrightRed     = "\033[101m";
        inline constexpr std::string_view kBrightGreen   = "\033[102m";
        inline constexpr std::string_view kBrightYellow  = "\033[103m";
        inline constexpr std::string_view kBrightBlue    = "\033[104m";
        inline constexpr std::string_view kBrightMagenta = "\033[105m";
        inline constexpr std::string_view kBrightCyan    = "\033[106m";
        inline constexpr std::string_view kBrightWhite   = "\033[107m";
    }

    /// ---------- Text attributes ----------
    namespace Styles {
        inline constexpr std::string_view kBold        = "\033[1m";
        inline constexpr std::string_view kDim         = "\033[2m";
        inline constexpr std::string_view kItalic      = "\033[3m";
        inline constexpr std::string_view kUnderline   = "\033[4m";
        inline constexpr std::string_view kBlink       = "\033[5m";
        inline constexpr std::string_view kInverse     = "\033[7m";
        inline constexpr std::string_view kHidden      = "\033[8m";
        inline constexpr std::string_view kStrike      = "\033[9m";

        inline constexpr std::string_view kNoBoldDim   = "\033[22m";
        inline constexpr std::string_view kNoItalic    = "\033[23m";
        inline constexpr std::string_view kNoUnderline = "\033[24m";
        inline constexpr std::string_view kNoBlink     = "\033[25m";
        inline constexpr std::string_view kNoInverse   = "\033[27m";
        inline constexpr std::string_view kNoHidden    = "\033[28m";
        inline constexpr std::string_view kNoStrike    = "\033[29m";
    }

    /// ---------- Symbols ----------
    namespace Symbols {
        inline constexpr std::string_view kArrowUp   = "▲";
        inline constexpr std::string_view kArrowDown = "▼";
        inline constexpr std::string_view kCheck     = "✔";
        inline constexpr std::string_view kCross     = "✘";
        inline constexpr std::string_view kDot       = "•";
        inline constexpr std::string_view kInfo      = "ℹ";
        inline constexpr std::string_view kWarn      = "⚠";
        inline constexpr std::string_view kClock     = "⏱";

        /// Heavy box drawing
        inline constexpr std::string_view kBoxTopLeft         = "┏";
        inline constexpr std::string_view kBoxTopSeparator    = "┳";
        inline constexpr std::string_view kBoxTopRight        = "┓";
        inline constexpr std::string_view kBoxMiddleLeft      = "┣";
        inline constexpr std::string_view kBoxMiddleSeparator = "╋";
        inline constexpr std::string_view kBoxMiddleRight     = "┫";
        inline constexpr std::string_view kBoxBottomLeft      = "┗";
        inline constexpr std::string_view kBoxBottomSeparator = "┻";
        inline constexpr std::string_view kBoxBottomRight     = "┛";
        inline constexpr std::string_view kBoxHorizontal      = "━";
        inline constexpr std::string_view kBoxVertical        = "┃";

        /// Rounded corners (use heavy horizontals)
        inline constexpr std::string_view kRoundedTopLeft     = "╭";
        inline constexpr std::string_view kRoundedTopRight    = "╮";
        inline constexpr std::string_view kRoundedBottomLeft  = "╰";
        inline constexpr std::string_view kRoundedBottomRight = "╯";
    }

    /// ---------- Control / Cursor ----------
    namespace Control {
        inline constexpr std::string_view kEraseToLineEnd   = "\033[K";
        inline constexpr std::string_view kEraseToLineStart = "\033[1K";
        inline constexpr std::string_view kEraseLine        = "\033[2K";

        inline constexpr std::string_view kClearScreen        = "\033[2J";
        inline constexpr std::string_view kClearScreenBelow   = "\033[0J";
        inline constexpr std::string_view kClearScreenAbove   = "\033[1J";
        inline constexpr std::string_view kHome               = "\033[H";

        inline constexpr std::string_view kSaveCursor    = "\033[s";
        inline constexpr std::string_view kRestoreCursor = "\033[u";
        inline constexpr std::string_view kHideCursor    = "\033[?25l";
        inline constexpr std::string_view kShowCursor    = "\033[?25h";
    }

    /// Parametric cursor movement
    inline std::string CursorUp(std::size_t n = 1)    { return "\033[" + std::to_string(n) + "A"; }
    inline std::string CursorDown(std::size_t n = 1)  { return "\033[" + std::to_string(n) + "B"; }
    inline std::string CursorRight(std::size_t n = 1) { return "\033[" + std::to_string(n) + "C"; }
    inline std::string CursorLeft(std::size_t n = 1)  { return "\033[" + std::to_string(n) + "D"; }
    inline std::string SetCursorPos(std::size_t row1, std::size_t col1) {
        return "\033[" + std::to_string(row1) + ";" + std::to_string(col1) + "H";
    }

    /// ---------- Color builders ----------
    inline std::string Fg256(std::uint8_t idx) { return "\033[38;5;" + std::to_string(idx) + "m"; }
    inline std::string Bg256(std::uint8_t idx) { return "\033[48;5;" + std::to_string(idx) + "m"; }
    inline std::string FgRGB(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
        return "\033[38;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m";
    }
    inline std::string BgRGB(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
        return "\033[48;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m";
    }

    /// ---------- Small helpers ----------
    inline std::string Repeat(std::string_view glyph, std::size_t count) {
        std::string s; s.reserve(glyph.size() * count);
        for (std::size_t i = 0; i < count; ++i) s.append(glyph);
        return s;
    }
    inline std::string ApplyColor(std::string_view s, std::string_view color) {
        std::string out; out.reserve(color.size() + s.size() + Colors::kReset.size());
        out.append(color).append(s).append(Colors::kReset);
        return out;
    }

    /// ---------- Bars and separators ----------
    enum class FrameStyle { Rounded, Box };

    /// Top bars:
    /// 1) Rounded:  ╭━━━━…╮
    /// 2) Box:      ┏━━━━…┓
    inline std::string TopBar(std::size_t inner_len,
                              std::string_view color,
                              FrameStyle style) {
        using namespace Symbols;
        const std::string_view left  = (style == FrameStyle::Rounded) ? kRoundedTopLeft : kBoxTopLeft;
        const std::string_view right = (style == FrameStyle::Rounded) ? kRoundedTopRight: kBoxTopRight;
        const std::string mid = Repeat(kBoxHorizontal, inner_len);
        return ApplyColor(std::string(left) + mid + std::string(right), color);
    }

    inline std::string TopBarRounded(std::size_t inner_len, std::string_view color) {
        return TopBar(inner_len, color, FrameStyle::Rounded);
    }
    inline std::string TopBarBox(std::size_t inner_len, std::string_view color) {
        return TopBar(inner_len, color, FrameStyle::Box);
    }

    /// Horizontal separators for tables.
    /// Spacings = widths of each segment between vertical junctions.
    /// kind: "top", "middle", or "bottom" row of a frame.
    enum class HSepKind { Top, Middle, Bottom };

    inline std::string HSeparator(const std::vector<std::size_t>& spacings,
                                  std::string_view color,
                                  FrameStyle style,
                                  HSepKind kind) {
        using namespace Symbols;

        /// Heavy glyph set
        std::string_view left;
        std::string_view midJunction;
        std::string_view right;
        switch (kind) {
            case HSepKind::Top:
                left  = (style == FrameStyle::Rounded) ? kRoundedTopLeft : kBoxTopLeft;
                midJunction = kBoxTopSeparator;   // heavy ┳
                right = (style == FrameStyle::Rounded) ? kRoundedTopRight : kBoxTopRight;
                break;
            case HSepKind::Middle:
                left  = kBoxMiddleLeft;           // ┣
                midJunction = kBoxMiddleSeparator;// ╋
                right = kBoxMiddleRight;          // ┫
                break;
            case HSepKind::Bottom:
                left  = kBoxBottomLeft;           // ┗
                midJunction = kBoxBottomSeparator;// ┻
                right = kBoxBottomRight;          // ┛
                break;
        }

        std::string out;
        out.reserve(16 + spacings.size() * 8);
        out.append(left);
        for (std::size_t i = 0; i < spacings.size(); ++i) {
            out.append(Repeat(kBoxHorizontal, spacings[i]));
            if (i + 1 < spacings.size()) out.append(midJunction);
        }
        out.append(right);
        return ApplyColor(out, color);
    }

    /// Convenience builders for each kind
    inline std::string HTop(const std::vector<std::size_t>& spacings,
                            std::string_view color,
                            FrameStyle style) {
        return HSeparator(spacings, color, style, HSepKind::Top);
    }
    inline std::string HMid(const std::vector<std::size_t>& spacings,
                            std::string_view color) {
        return HSeparator(spacings, color, FrameStyle::Box, HSepKind::Middle);
    }
    inline std::string HBottom(const std::vector<std::size_t>& spacings,
                               std::string_view color,
                               FrameStyle style = FrameStyle::Box) {
        return HSeparator(spacings, color, style, HSepKind::Bottom);
    }

    /// Pack lines as an array if you want to emit a header frame quickly.
    /// Example: BuildTopHeaderArray(12, Colors::kBrightYellow, FrameStyle::Rounded)
    inline std::vector<std::string> BuildTopHeaderArray(std::size_t inner_len,
                                                        std::string_view color,
                                                        FrameStyle style) {
        /// Single-line array as requested, ready to print or concatenate.
        return { TopBar(inner_len, color, style) };
    }







    namespace Nott::Plot::Details::Reliability::detail {
        inline auto PickColor(std::size_t index) -> std::string
        {
            static constexpr std::array<const char*, 8> palette{
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f"
            };
            return std::string(palette[index % palette.size()]);
        }
    }
}


/// Usage example:
///
/// @code
/// using namespace Nott::Utils::Terminal;
///
/// // Top bars
/// auto r = TopBarRounded(13, Colors::kBrightYellow);
/// auto b = TopBarBox(13, Colors::kBrightYellow);
///
/// // Spacing vector [6, 4, 8]
/// std::vector<std::size_t> spans{6,4,8};
/// auto top = HTop(spans, Colors::kBrightGreen, FrameStyle::Box);
/// auto mid = HMid(spans, Colors::kBrightGreen);
/// auto bot = HBottom(spans, Colors::kBrightGreen, FrameStyle::Box);
/// @endcode
#endif // LIBNott_TERMINAL_HPP
