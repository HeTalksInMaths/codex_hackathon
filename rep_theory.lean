import Lean

open Lean

structure Config where
  decls : Array Name := #[]
  depth : Nat := 1
  namespaceFilters : Array String := #[]
  rootModule : Name := `Mathlib.RepresentationTheory.Basic
  deriving Repr

partial def collectConsts (e : Expr) (acc : Std.HashSet Name := {}) : Std.HashSet Name :=
  match e with
  | .const n _ => acc.insert n
  | .app f a => collectConsts a (collectConsts f acc)
  | .lam _ t b _ => collectConsts b (collectConsts t acc)
  | .forallE _ t b _ => collectConsts b (collectConsts t acc)
  | .letE _ t v b _ => collectConsts b (collectConsts v (collectConsts t acc))
  | .mdata _ b => collectConsts b acc
  | .proj _ _ b => collectConsts b acc
  | _ => acc

def constInfoExprs (ci : ConstantInfo) : Array Expr :=
  match ci with
  | .axiomInfo i => #[i.type]
  | .thmInfo i => #[i.type, i.value]
  | .opaqueInfo i => #[i.type, i.value]
  | .defnInfo i => #[i.type, i.value]
  | .quotInfo i => #[i.type]
  | .inductInfo i => #[i.type]
  | .ctorInfo i => #[i.type]
  | .recInfo i => #[i.type]

def parseName (s : String) : Name :=
  (s.splitOn ".").foldl (fun n part => Name.str n part) Name.anonymous

def allowedByNamespace (filters : Array String) (n : Name) : Bool :=
  if filters.isEmpty then
    true
  else
    filters.any fun f => n.toString.startsWith f

def nameLt (a b : Name) : Bool :=
  Name.quickCmp a b == Ordering.lt

def directDeps (env : Environment) (filters : Array String) (n : Name) : IO (Array Name) := do
  match env.find? n with
  | none =>
      IO.eprintln s!"warning: declaration not found: {n}"
      pure #[]
  | some ci =>
      let mut deps : Std.HashSet Name := {}
      for e in constInfoExprs ci do
        deps := collectConsts e deps
      let mut out : Array Name := #[]
      for dep in deps.toArray do
        if dep != n && env.contains dep && allowedByNamespace filters dep then
          out := out.push dep
      pure <| out.qsort nameLt

partial def buildGraph
    (env : Environment)
    (filters : Array String)
    (frontier : Array Name)
    (depth : Nat)
    (seen : Std.HashSet Name := {})
    (edges : Array (Name × Name) := #[]) : IO (Std.HashSet Name × Array (Name × Name)) := do
  if depth == 0 || frontier.isEmpty then
    pure (seen, edges)
  else
    let mut newSeen := seen
    let mut newEdges := edges
    let mut next : Array Name := #[]
    for src in frontier do
      if !newSeen.contains src then
        newSeen := newSeen.insert src
      let deps ← directDeps env filters src
      for dst in deps do
        newEdges := newEdges.push (src, dst)
        if !newSeen.contains dst then
          next := next.push dst
    buildGraph env filters next depth.succ.pred newSeen newEdges

partial def parseArgs : List String → Config → IO Config
  | [], cfg => pure cfg
  | "--decl" :: d :: rest, cfg =>
      parseArgs rest { cfg with decls := cfg.decls.push (parseName d) }
  | "--module" :: m :: rest, cfg =>
      parseArgs rest { cfg with rootModule := parseName m }
  | "--depth" :: d :: rest, cfg =>
      match d.toNat? with
      | some n => parseArgs rest { cfg with depth := n }
      | none => throw <| IO.userError s!"invalid --depth value: {d}"
  | "--namespace" :: ns :: rest, cfg =>
      parseArgs rest { cfg with namespaceFilters := cfg.namespaceFilters.push ns }
  | arg :: _, _ =>
      throw <| IO.userError s!"unknown argument: {arg}\nUsage: lake env lean --run rep_theory.lean --decl <Name> [--decl <Name> ...] [--module Module.Name] [--depth N] [--namespace Prefix]"

def renderDot (nodes : Std.HashSet Name) (edges : Array (Name × Name)) : String := Id.run do
  let mut lines := #["digraph RepTheory {"]
  let sortedNodes := nodes.toArray.qsort nameLt
  for n in sortedNodes do
    lines := lines.push s!"  \"{n}\";"
  for (src, dst) in edges do
    lines := lines.push s!"  \"{src}\" -> \"{dst}\";"
  lines := lines.push "}"
  String.intercalate "\n" lines.toList

def main (args : List String) : IO Unit := do
  let cfg ← parseArgs args {}
  if cfg.decls.isEmpty then
    throw <| IO.userError "Provide at least one --decl <Name>."

  let env ← importModules #[{ module := cfg.rootModule }] {}
  let (nodes, edges) ← buildGraph env cfg.namespaceFilters cfg.decls cfg.depth
  IO.println (renderDot nodes edges)
