# Third-party dependencies

## ForkUnion

- Project: `ashvardanian/ForkUnion`
- Version: `3.0.2`
- Commit: `9f1d91c3529fc735f0ab60cbc895f449c7139d28`
- License: Apache-2.0
- Archive SHA-256:
  `149e7a443e50f157847c8614596341e12c9c98c6ba8287dc6996e5554ebd3bcf`

The release archive is fetched by CMake and used header-only. Fast-math uses a
thread-local pool cache so repeated bulk-synchronous kernels avoid creating
and joining fresh worker threads while concurrent callers retain independent
pools.

## nauty

- Project: nauty and Traces
- Version tested: `2.9.3`
- License: Apache-2.0
- Discovery: optional system library through `pkg-config libnauty`

Fast-math links nauty only when it is already installed. The native boundary
uses nauty's dense directed-graph API for batched vertex-colored canonical
labels and automorphism generators; builds without nauty retain the exhaustive
small-order Python reference backend.
