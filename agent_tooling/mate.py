@register_tool(description="Extract basic contact information from text")
def extract_contact_info(action_context: ActionContext, text: str) -> dict:
    """Extract name, email, and phone from text using a smaller, faster model."""
    # Use a smaller model for simple extraction
    response = action_context.get("fast_llm")(
        Prompt(
            messages=[
                {
                    "role": "system",
                    "content": "Extract contact information in JSON format.",
                },
                {"role": "user", "content": text},
            ]
        )
    )
    return json.loads(response)


@register_tool(description="Analyze complex technical documentation")
def analyze_technical_doc(action_context: ActionContext, document: str) -> dict:
    """Perform deep analysis of technical documentation."""
    # Use a more capable model for complex analysis
    response = action_context.get("powerful_llm")(
        Prompt(
            messages=[
                {
                    "role": "system",
                    "content": "Analyze technical this documentation thoroughly to identify potential contradictions in process that could lead to unexpected problems.",
                },
                {"role": "user", "content": document},
            ]
        )
    )
    return json.loads(response)


# Too generic - opens up possibilities for misuse
@register_tool(description="Modify calendar events")
def update_calendar(
    action_context: ActionContext, event_id: str, updates: dict
) -> dict:
    """Update any aspect of a calendar event."""
    return calendar.update_event(event_id, updates)


# More specific - clear purpose and limited scope
@register_tool(description="Reschedule a meeting you own to a new time")
def reschedule_my_meeting(
    action_context: ActionContext,
    event_id: str,
    new_start_time: str,
    new_duration_minutes: int,
) -> dict:
    """
    Reschedule a meeting you own to a new time.
    Only works for meetings where you are the organizer.
    """
    # Verify ownership
    event = calendar.get_event(event_id)
    if event.organizer != action_context.get("user_email"):
        raise ValueError("Can only reschedule meetings you organize")

    # Validate new time is in the future
    new_start = datetime.fromisoformat(new_start_time)
    if new_start < datetime.now():
        raise ValueError("Cannot schedule meetings in the past")

    return calendar.update_event_time(
        event_id, new_start_time=new_start_time, duration_minutes=new_duration_minutes
    )


# Token inefficient - includes unnecessary context
@register_tool(description="Analyze sales data to identify trends and patterns...")
def analyze_sales(action_context: ActionContext, data: str) -> str:
    """
    This function will analyze sales data to identify trends and patterns.
    It looks at various aspects including:
    - Monthly trends
    - Seasonal patterns
    - Year-over-year growth
    - Product category performance
    - Regional variations
    - Customer segments

    The analysis will be thorough and consider multiple factors...
    [More verbose documentation]
    """

    # This prompt asks for unneeded analysis, leading to both a waste of input and output tokens
    return prompt_llm(
        action_context,
        f"""
        Analyze this sales data thoroughly. Consider monthly trends,
        seasonal patterns, year-over-year growth, product categories,
        regional variations, and customer segments. Provide detailed
        insights about all these aspects.

        Data: {data}

        Please give a comprehensive analysis...
    """,
    )


# Token efficient - focused and precise
@register_tool(description="Analyze sales data for key trends")
def analyze_sales(action_context: ActionContext, data: str) -> str:
    """Calculate key sales metrics and identify significant trends."""

    # This prompt is focused and precise, using tokens efficiently
    return prompt_llm(
        action_context,
        f"""
        Sales Data: {data}
        1. Calculate YoY growth
        2. Identify top 3 trends
        3. Flag significant anomalies
    """,
    )


class ReversibleAction:
    def __init__(self, execute_func, reverse_func):
        self.execute = execute_func
        self.reverse = reverse_func
        self.execution_record = None

    def run(self, **args):
        """Execute action and record how to reverse it."""
        result = self.execute(**args)
        self.execution_record = {
            "args": args,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }
        return result

    def undo(self):
        """Reverse the action using recorded information."""
        if not self.execution_record:
            raise ValueError("No action to reverse")
        return self.reverse(**self.execution_record)


# Example using reversible actions
create_event = ReversibleAction(
    execute_func=calendar.create_event,
    reverse_func=lambda **record: calendar.delete_event(record["result"]["event_id"]),
)

send_invite = ReversibleAction(
    execute_func=calendar.send_invite,
    reverse_func=lambda **record: calendar.cancel_invite(record["result"]["invite_id"]),
)


class ActionTransaction:
    def __init__(self):
        self.actions = []
        self.executed = []
        self.committed = False
        self.transaction_id = str(uuid.uuid4())

    def add(self, action: ReversibleAction, **args):
        """Queue an action for execution."""
        if self.committed:
            raise ValueError("Transaction already committed")
        self.actions.append((action, args))

    async def execute(self):
        """Execute all actions in the transaction."""
        try:
            for action, args in self.actions:
                result = action.run(**args)
                self.executed.append(action)
        except Exception as e:
            # If any action fails, reverse everything done so far
            await self.rollback()
            raise e

    async def rollback(self):
        """Reverse all executed actions in reverse order."""
        for action in reversed(self.executed):
            await action.undo()
        self.executed = []

    def commit(self):
        """Mark transaction as committed."""
        self.committed = True


class StagedActionEnvironment(Environment):
    def __init__(self):
        self.staged_transactions = {}
        self.llm = None  # High-capability LLM for review

    def stage_actions(self, task_id: str) -> ActionTransaction:
        """Create a new transaction for staging actions."""
        transaction = ActionTransaction()
        self.staged_transactions[task_id] = transaction
        return transaction

    def review_transaction(self, task_id: str) -> bool:
        """Have LLM review staged actions for safety."""
        transaction = self.staged_transactions.get(task_id)
        if not transaction:
            raise ValueError(f"No transaction found for task {task_id}")

        # Create a description of staged actions
        staged_actions = [
            f"Action: {action.__class__.__name__}\nArgs: {args}"
            for action, args in transaction.actions
        ]

        # The safest way to do this would be to send it for human review, but we can also imagine having a more capable AI system review it before the human to minimize the number of reviews that the human has to do. The more capable AI can review and reject potentially problematic actions earlier.

        review_prompt = f"""Review these staged actions for safety:

        Task ID: {task_id}

        Staged Actions:
        {staged_actions}

        Consider:
        1. Are all actions necessary for the task?
        2. Could any action have unintended consequences?
        3. Are the actions in a safe order?
        4. Is there a safer way to achieve the same goal?

        Should these actions be approved?
        """

        response = self.llm.generate(review_prompt)

        # If approved, notify the human and ask if
        # they want to proceed
        return "approved" in response.lower()


# Example usage:
async def schedule_team_meeting(
    env: StagedActionEnvironment, attendees: List[str], duration: int
):
    """Schedule a team meeting with safety checks."""
    task_id = str(uuid.uuid4())
    transaction = env.stage_actions(task_id)

    # Check availability (execute immediately)
    available_slots = calendar.check_availability(attendees, duration)
    if not available_slots:
        return {"error": "No available time slots"}

    best_slot = available_slots[0]

    # Stage the event creation
    transaction.add(
        create_event, title="Team Meeting", time=best_slot, duration=duration
    )

    # Draft email (execute immediately)
    email_draft = email.draft_message(
        to=attendees,
        subject="Team Meeting",
        body=f"Team meeting scheduled for {best_slot}",
    )

    # Stage the email send
    transaction.add(send_email, draft_id=email_draft.id)

    # Review staged actions...send to human review
    # or more capable AI for initial filtering
    if env.review_transaction(task_id):
        await transaction.execute()
        transaction.commit()
        return {"status": "scheduled"}
    else:
        return {"status": "rejected"}


# Approach 1: Multiple loosely constrained tools
@register_tool(description="Create a calendar event")
def create_calendar_event(
    action_context: ActionContext, title: str, time: str, attendees: List[str]
) -> dict:
    """Create a calendar event."""
    return calendar.create_event(title=title, time=time, attendees=attendees)


@register_tool(description="Send email to attendees")
def send_email(
    action_context: ActionContext, to: List[str], subject: str, body: str
) -> dict:
    """Send an email."""
    return email.send(to=to, subject=subject, body=body)


@register_tool(description="Update calendar event")
def update_event(action_context: ActionContext, event_id: str, updates: dict) -> dict:
    """Update any aspect of a calendar event."""
    return calendar.update_event(event_id, updates)


# Approach 2: Single comprehensive safe tool
@register_tool(description="Schedule a team meeting safely")
def schedule_team_meeting(
    action_context: ActionContext,
    title: str,
    description: str,
    attendees: List[str],
    duration_minutes: int,
    timeframe: str = "next_week",
) -> dict:
    """
    Safely schedule a team meeting with all necessary coordination.

    This tool:
    1. Verifies all attendees are valid
    2. Checks calendar availability
    3. Creates the event at the best available time
    4. Sends appropriate notifications
    5. Handles all error cases
    """
    # Input validation
    if not 15 <= duration_minutes <= 120:
        raise ValueError("Meeting duration must be between 15 and 120 minutes")

    if len(attendees) > 10:
        raise ValueError("Cannot schedule meetings with more than 10 attendees")

    # Verify attendees
    valid_attendees = validate_attendees(attendees)
    if len(valid_attendees) != len(attendees):
        raise ValueError("Some attendees are invalid")

    # Find available times
    available_slots = find_available_times(
        attendees=valid_attendees, duration=duration_minutes, timeframe=timeframe
    )

    if not available_slots:
        return {"status": "no_availability", "message": "No suitable time slots found"}

    # Create event at best time
    event = calendar.create_event(
        title=title,
        description=description,
        time=available_slots[0],
        duration=duration_minutes,
        attendees=valid_attendees,
    )

    # Send notifications
    notifications.send_meeting_scheduled(event_id=event.id, attendees=valid_attendees)

    return {
        "status": "scheduled",
        "event_id": event.id,
        "scheduled_time": available_slots[0],
    }
