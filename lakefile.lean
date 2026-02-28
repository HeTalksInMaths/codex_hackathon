import Lake
open Lake DSL

package «codex_hackathon» where
  -- add package configuration options here

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «CodexHackathon» where
  -- add library configuration options here
