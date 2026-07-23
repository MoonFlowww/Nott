#ifndef Nott_LIBRARY_H
#define Nott_LIBRARY_H

#include "../src/core/core.hpp"
#include "../src/layer/layer.hpp"
#include "../src/loss/loss.hpp"
#include "../src/optimizer/optimizer.hpp"

#include "../src/attention/attention.hpp"
#include "../src/block/block.hpp"
#include "../src/data/data.hpp"



// Public umbrella header.
// -----------------------------------------------------------------------------
// Intention:
//  - Re-export the minimal API surface required by downstream applications
//    (core orchestrator entry points + pure network handles).
//  - Include-only components; implementation lives in header-only modules under
//    src/ to guarantee compile-time composition and zero-cost abstractions.
//  - Provide a single spot to toggle build flags (concept checks, static
//    assertions, compile-time diagnostics) once the runtime skeleton is ready.

#endif // Nott_LIBRARY_H