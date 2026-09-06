# Counter matched hypernetwork baselines

All three controls use the same recurrent policy encoder, Transformer relation
encoder, two-layer generated Q head, QMIX learner, condition width, seed and
training schedule. The Transformer representation is always the Q-head input;
only the input that generates the Q-head parameters changes. They do not
use a learned mask, random drop, KL regularisation, or relation auxiliary loss.
Only the input used to generate the Q-head parameters changes.

- `hyper_hypermarl_id`: one-hot agent ID, projected to the shared condition
  width. The generated head uses the existing HyperMARL-style initialisation.
- `hyper_cash_obs_type`: current local observation concatenated with a one-hot
  tactical player type. Counter defaults to `0,1,2,2`; override
  `clean_counter_agent_types` when a different role ordering is intended.
- `hyper_rpg_relation`: current local observation is encoded into a relation
  condition, and that condition generates the head. This is the
  single-task RPG-inspired comparison; it does not claim to reproduce RPG's
  continual-learning task embeddings or regularisers.

The explicit role list is intentional: the current 30-dimensional Counter
wrapper omits GRF's raw `left_team_roles`, so silently treating agent ID as
player type would make the CASH and HyperMARL controls equivalent.
