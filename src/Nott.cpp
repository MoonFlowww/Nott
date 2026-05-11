/// Entry point bridge for the library (to be linked with user application).
/// -----------------------------------------------------------------------------
/// Planned workflow:
///  - Parse CLI / configuration files and materialise a constexpr
///    Nott::Core::CompileTimeConfig instance (likely via constexpr builders).
///  - Call Nott::Core::make_runtime() to receive an aggregated facade exposing
///    dataset loaders, network handles and training/evaluation pipelines.
///  - Launch training/evaluation routines while measuring latency (no dynamic
///    feature toggles -- decisions already compiled into the runtime facade).
///  - Provide hooks for stress testing utilities under src/utils/ and produce
///    analytics for quantitative research workflows.