#ifndef Nott_TESTS_PRELUDE_HPP
#define Nott_TESTS_PRELUDE_HPP
/// Include this instead of doctest.h directly.
///
/// torch/torch.h pulls in c10/util/logging_is_not_google_glog.h, which defines
/// CHECK unguarded as a fatal glog-style assert. Every test file included
/// doctest first and torch second, so that definition silently won and the
/// suite's CHECKs were aborting the process on failure instead of reporting
/// one: no expected-vs-actual values, no assertion counted, and every test
/// after the first failure in that binary never ran. Pulling torch in first and
/// dropping its CHECK here means tests get doctest's.
#include <torch/torch.h>

#ifdef CHECK
#undef CHECK
#endif

#include "third_party/doctest.h"

#endif // Nott_TESTS_PRELUDE_HPP
