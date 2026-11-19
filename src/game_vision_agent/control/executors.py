from game_vision_agent.interfaces import ActionOutcome, ActionProposal


class NoOpExecutor:
    def execute(self, proposal: ActionProposal, *, simulator: bool = False) -> ActionOutcome:
        # This phase never sends input to a desktop or game. Simulator contract is reserved.
        return ActionOutcome(proposal.action_id, executed=False,
                            recovery_decision="no executor is enabled in Phase 1")
