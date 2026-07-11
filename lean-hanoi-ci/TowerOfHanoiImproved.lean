import Mathlib

namespace TowerOfHanoi

/-- The three pegs used by the Tower of Hanoi puzzle. -/
inductive Peg where
  | A | B | C
  deriving DecidableEq, Repr

/-- A single move names the disk and its source and destination pegs.
    Disks are numbered from 1 (smallest) upward. -/
structure Move where
  disk : Nat
  src  : Peg
  dst  : Peg
  deriving DecidableEq, Repr

/-- A compressed plan.  `seq p q` means execute `p`, then execute `q`. -/
inductive Plan where
  | stop
  | one (m : Move)
  | seq (p q : Plan)
  deriving DecidableEq, Repr

/-- Number of primitive disk moves represented by a plan. -/
def moveCount : Plan → Nat
  | .stop    => 0
  | .one _   => 1
  | .seq p q => moveCount p + moveCount q

/-- The standard recursive Hanoi plan. -/
def hanoi : Nat → Peg → Peg → Peg → Plan
  | 0,     _,   _,   _   => .stop
  | n + 1, src, dst, aux =>
      .seq (hanoi n src aux dst)
        (.seq (.one { disk := n + 1, src := src, dst := dst })
          (hanoi n aux dst src))

/-- The pegs in a recursive call must be pairwise distinct. -/
def Distinct3 (x y z : Peg) : Prop :=
  x ≠ y ∧ x ≠ z ∧ y ≠ z

lemma distinct_src_aux_dst {src dst aux : Peg}
    (h : Distinct3 src dst aux) : Distinct3 src aux dst := by
  rcases h with ⟨hSD, hSA, hDA⟩
  exact ⟨hSA, hSD, Ne.symm hDA⟩

lemma distinct_aux_dst_src {src dst aux : Peg}
    (h : Distinct3 src dst aux) : Distinct3 aux dst src := by
  rcases h with ⟨hSD, hSA, hDA⟩
  exact ⟨Ne.symm hDA, Ne.symm hSA, Ne.symm hSD⟩

/-- Structural certificate for the standard recursive decomposition. -/
inductive Solves : Nat → Peg → Peg → Peg → Plan → Prop where
  | zero {src dst aux : Peg}
      (hDistinct : Distinct3 src dst aux) :
      Solves 0 src dst aux .stop
  | step {n : Nat} {src dst aux : Peg} {left right : Plan}
      (hDistinct : Distinct3 src dst aux)
      (hLeft  : Solves n src aux dst left)
      (hRight : Solves n aux dst src right) :
      Solves (n + 1) src dst aux
        (.seq left
          (.seq (.one { disk := n + 1, src := src, dst := dst }) right))

/-- The generated plan has a structural certificate for every `n`. -/
theorem hanoi_solves :
    ∀ (n : Nat) (src dst aux : Peg),
      Distinct3 src dst aux → Solves n src dst aux (hanoi n src dst aux) := by
  intro n
  induction n with
  | zero =>
      intro src dst aux h
      simpa [hanoi] using Solves.zero h
  | succ n ih =>
      intro src dst aux h
      simpa [hanoi] using
        Solves.step h
          (ih src aux dst (distinct_src_aux_dst h))
          (ih aux dst src (distinct_aux_dst_src h))

/-- A concrete physical state. Each list is stored top-to-bottom. -/
structure Stacks where
  a : List Nat
  b : List Nat
  c : List Nat
  deriving DecidableEq, Repr

namespace Stacks

/-- Read one peg from a physical state. -/
def get (s : Stacks) : Peg → List Nat
  | .A => s.a
  | .B => s.b
  | .C => s.c

/-- Replace one peg in a physical state. -/
def set (s : Stacks) (p : Peg) (xs : List Nat) : Stacks :=
  match p with
  | .A => { s with a := xs }
  | .B => { s with b := xs }
  | .C => { s with c := xs }

@[simp] theorem get_set_same (s : Stacks) (p : Peg) (xs : List Nat) :
    (s.set p xs).get p = xs := by
  cases p <;> rfl

@[simp] theorem get_set_of_ne (s : Stacks) {p q : Peg} (h : q ≠ p)
    (xs : List Nat) : (s.set p xs).get q = s.get q := by
  cases p <;> cases q <;> simp_all [set, get]

@[simp] theorem set_set_same (s : Stacks) (p : Peg)
    (xs ys : List Nat) : (s.set p xs).set p ys = s.set p ys := by
  cases p <;> rfl

end Stacks

/-- `[1,2,...,n]`, with disk 1 at the top. -/
def orderedTower (n : Nat) : List Nat :=
  (List.range n).map (fun k => k + 1)

lemma orderedTower_succ (n : Nat) :
    orderedTower (n + 1) = orderedTower n ++ [n + 1] := by
  simp [orderedTower, List.range_succ]

/-- All `n` disks on peg `p`, with other pegs empty. -/
def towerState (n : Nat) (p : Peg) : Stacks :=
  (Stacks.mk [] [] []).set p (orderedTower n)

/-- Execute one move while checking both physical Hanoi rules. -/
def checkedMove (s : Stacks) (m : Move) : Option Stacks :=
  match s.get m.src with
  | [] => none
  | d :: sourceTail =>
      if d ≠ m.disk then
        none
      else
        let targetStack := s.get m.dst
        match targetStack with
        | [] =>
            let s' := s.set m.src sourceTail
            some (s'.set m.dst [d])
        | top :: _ =>
            if d < top then
              let s' := s.set m.src sourceTail
              some (s'.set m.dst (d :: targetStack))
            else
              none

/-- Execute a compressed plan, failing at the first illegal move. -/
def runChecked : Stacks → Plan → Option Stacks
  | s, .stop    => some s
  | s, .one m   => checkedMove s m
  | s, .seq p q =>
      match runChecked s p with
      | none    => none
      | some s' => runChecked s' q

/-! ## Generic operational bridge

The key strengthening is to run a small tower on top of an arbitrary `base`
whose disks are all larger. This is the induction invariant needed by the
recursive algorithm: during a subproblem, the untouched larger disks form the
base underneath the moving tower.
-/

/-- Put disks `1,...,n` on top of peg `p` in an existing state. -/
def putTower (n : Nat) (p : Peg) (base : Stacks) : Stacks :=
  base.set p (orderedTower n ++ base.get p)

/-- Push one named disk onto a peg of a base state. -/
def pushDisk (d : Nat) (p : Peg) (base : Stacks) : Stacks :=
  base.set p (d :: base.get p)

/-- Every disk in the base is strictly larger than every disk `≤ n`. -/
def BaseLargerThan (n : Nat) (base : Stacks) : Prop :=
  ∀ p d, d ∈ base.get p → n < d

lemma putTower_succ (n : Nat) (p : Peg) (base : Stacks) :
    putTower (n + 1) p base = putTower n p (pushDisk (n + 1) p base) := by
  simp [putTower, pushDisk, orderedTower_succ, List.append_assoc]

lemma pushDisk_baseLarger {n : Nat} {p : Peg} {base : Stacks}
    (h : BaseLargerThan (n + 1) base) :
    BaseLargerThan n (pushDisk (n + 1) p base) := by
  intro q d hd
  by_cases hqp : q = p
  · subst q
    simp [pushDisk] at hd
    rcases hd with rfl | hd
    · omega
    · have := h p d hd
      omega
  · have hd' : d ∈ base.get q := by
      simpa [pushDisk, Stacks.get_set_of_ne _ hqp] using hd
    have := h q d hd'
    omega

/-- A one-step executor lemma, independent of Hanoi. -/
lemma checkedMove_of_top
    (s : Stacks) (d : Nat) (src dst : Peg) (sourceTail target : List Nat)
    (hsrc : s.get src = d :: sourceTail)
    (hdst : s.get dst = target)
    (hplace : match target with | [] => True | top :: _ => d < top) :
    checkedMove s { disk := d, src := src, dst := dst } =
      some ((s.set src sourceTail).set dst (d :: target)) := by
  cases target with
  | nil =>
      simp [checkedMove, hsrc, hdst]
  | cons top rest =>
      simp [checkedMove, hsrc, hdst] at hplace ⊢
      exact hplace

/-- Moving disk `n+1` between two distinct pegs, while the smaller tower sits
on the third peg, updates only the larger-disk base. -/
lemma checkedMove_largest
    (n : Nat) (src dst aux : Peg) (base : Stacks)
    (hDistinct : Distinct3 src dst aux)
    (hLarge : BaseLargerThan (n + 1) base) :
    checkedMove
      (putTower n aux (pushDisk (n + 1) src base))
      { disk := n + 1, src := src, dst := dst }
      = some (putTower n aux (pushDisk (n + 1) dst base)) := by
  rcases hDistinct with ⟨hSD, hSA, hDA⟩
  have hSource :
      (putTower n aux (pushDisk (n + 1) src base)).get src =
        (n + 1) :: base.get src := by
    simp [putTower, pushDisk, hSA]
  have hTarget :
      (putTower n aux (pushDisk (n + 1) src base)).get dst = base.get dst := by
    simp [putTower, pushDisk, hSD, Ne.symm hDA]
  have hPlace :
      match base.get dst with
      | [] => True
      | top :: _ => n + 1 < top := by
    cases hdst : base.get dst with
    | nil => trivial
    | cons top rest =>
        exact hLarge dst top (by simp [hdst])
  rw [checkedMove_of_top
    (s := putTower n aux (pushDisk (n + 1) src base))
    (d := n + 1) (src := src) (dst := dst)
    (sourceTail := base.get src) (target := base.get dst)
    hSource hTarget hPlace]
  congr 1
  ext p
  cases p <;> cases src <;> cases dst <;> cases aux <;>
    simp_all [Distinct3, putTower, pushDisk, Stacks.get, Stacks.set]

/-- **Generic operational correctness.** The recursive Hanoi plan legally
moves the small tower over any untouched base of larger disks. -/
theorem hanoi_correct_over_base :
    ∀ (n : Nat) (src dst aux : Peg) (base : Stacks),
      Distinct3 src dst aux →
      BaseLargerThan n base →
      runChecked (putTower n src base) (hanoi n src dst aux) =
        some (putTower n dst base) := by
  intro n
  induction n with
  | zero =>
      intro src dst aux base hDistinct hLarge
      simp [hanoi, runChecked, putTower, orderedTower]
  | succ n ih =>
      intro src dst aux base hDistinct hLarge
      rw [putTower_succ]
      simp only [hanoi, runChecked]
      rw [ih src aux dst (pushDisk (n + 1) src base)
        (distinct_src_aux_dst hDistinct)
        (pushDisk_baseLarger hLarge)]
      rw [checkedMove_largest n src dst aux base hDistinct hLarge]
      rw [ih aux dst src (pushDisk (n + 1) dst base)
        (distinct_aux_dst_src hDistinct)
        (pushDisk_baseLarger hLarge)]
      exact congrArg some (putTower_succ n dst base).symm

/-- Empty physical state. -/
def emptyStacks : Stacks := ⟨[], [], []⟩

lemma emptyStacks_baseLarger (n : Nat) : BaseLargerThan n emptyStacks := by
  intro p d hd
  cases p <;> simp [emptyStacks, Stacks.get] at hd

lemma putTower_empty (n : Nat) (p : Peg) :
    putTower n p emptyStacks = towerState n p := by
  cases p <;> simp [putTower, towerState, emptyStacks, Stacks.get, Stacks.set]

/-- The desired bridge for ordinary Hanoi states, for every number of disks. -/
theorem hanoi_operationally_correct
    (n : Nat) (src dst aux : Peg)
    (hDistinct : Distinct3 src dst aux) :
    runChecked (towerState n src) (hanoi n src dst aux) =
      some (towerState n dst) := by
  simpa [putTower_empty] using
    hanoi_correct_over_base n src dst aux emptyStacks hDistinct
      (emptyStacks_baseLarger n)

/-- Operational meaning of a compressed plan in every larger-disk context. -/
def OperationallySolves
    (n : Nat) (src dst aux : Peg) (p : Plan) : Prop :=
  Distinct3 src dst aux ∧
  ∀ base, BaseLargerThan n base →
    runChecked (putTower n src base) p = some (putTower n dst base)

/-- The generated plan has both its old structural certificate and a true
physical execution certificate. -/
theorem hanoi_operationallySolves
    (n : Nat) (src dst aux : Peg)
    (hDistinct : Distinct3 src dst aux) :
    OperationallySolves n src dst aux (hanoi n src dst aux) := by
  refine ⟨hDistinct, ?_⟩
  intro base hLarge
  exact hanoi_correct_over_base n src dst aux base hDistinct hLarge

/-- Move-count recurrence. -/
def expectedMoves : Nat → Nat
  | 0     => 0
  | n + 1 => 2 * expectedMoves n + 1

theorem moveCount_hanoi (n : Nat) (src dst aux : Peg) :
    moveCount (hanoi n src dst aux) = expectedMoves n := by
  induction n generalizing src dst aux with
  | zero => simp [hanoi, moveCount, expectedMoves]
  | succ n ih =>
      simp [hanoi, moveCount, expectedMoves, ih]
      omega

theorem expectedMoves_add_one (n : Nat) : expectedMoves n + 1 = 2 ^ n := by
  induction n with
  | zero => simp [expectedMoves]
  | succ n ih =>
      calc
        expectedMoves (n + 1) + 1 = 2 * (expectedMoves n + 1) := by
          simp [expectedMoves]
          omega
        _ = 2 * (2 ^ n) := by rw [ih]
        _ = 2 ^ (n + 1) := by simp [Nat.pow_succ, Nat.mul_comm]

theorem expectedMoves_closed (n : Nat) : expectedMoves n = 2 ^ n - 1 := by
  have h := expectedMoves_add_one n
  omega

/-- Regression instance. -/
def plan20 : Plan := hanoi 20 .A .C .B

theorem plan20_solves : Solves 20 .A .C .B plan20 := by
  change Solves 20 .A .C .B (hanoi 20 .A .C .B)
  exact hanoi_solves 20 .A .C .B (by decide)

theorem plan20_moveCount : moveCount plan20 = 1048575 := by
  change moveCount (hanoi 20 .A .C .B) = 1048575
  rw [moveCount_hanoi, expectedMoves_closed]
  norm_num

/-- The old million-move computation now follows immediately from the generic
symbolic theorem; no million-step reduction is needed for the proof. -/
theorem plan20_operationally_valid :
    runChecked (towerState 20 .A) plan20 = some (towerState 20 .C) := by
  exact hanoi_operationally_correct 20 .A .C .B (by decide)

end TowerOfHanoi
