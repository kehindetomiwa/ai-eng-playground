"""
Act as Persona X
Perform task Y
"""

@register_tool()
def prompt_expert(
    action_context: ActionContext, description_of_expert: str, prompt: str
) -> str:
    """
    Generate a response from an expert persona.

    The expert's background and specialization should be thoroughly described to ensure
    responses align with their expertise. The prompt should be focused on topics within
    their domain of knowledge.

    Args:
        description_of_expert: Detailed description of the expert's background and expertise
        prompt: The specific question or task for the expert

    Returns:
        The expert's response
    """
    generate_response = action_context.get("llm")
    response = generate_response(
        Prompt(
            messages=[
                {
                    "role": "system",
                    "content": f"Act as the following expert and respond accordingly: {description_of_expert}",
                },
                {"role": "user", "content": prompt},
            ]
        )
    )
    return response


@register_tool(tags=["documentation"])
def generate_technical_documentation(
    action_context: ActionContext, code_or_feature: str
) -> str:
    """
    Generate technical documentation by consulting a senior technical writer.
    This expert focuses on creating clear, comprehensive documentation for developers.

    Args:
        code_or_feature: The code or feature to document
    """
    return prompt_expert(
        action_context=action_context,
        description_of_expert="""
        You are a senior technical writer with 15 years of experience in software documentation.
        You have particular expertise in:
        - Writing clear and precise API documentation
        - Explaining complex technical concepts to developers
        - Documenting implementation details and integration points
        - Creating code examples that illustrate key concepts
        - Identifying and documenting important caveats and edge cases

        Your documentation is known for striking the perfect balance between completeness
        and clarity. You understand that good technical documentation serves as both
        a reference and a learning tool.
        """,
        prompt=f"""
        Please create comprehensive technical documentation for the following code or feature:

        {code_or_feature}

        Your documentation should include:
        1. A clear overview of the feature's purpose and functionality
        2. Detailed explanation of the implementation approach
        3. Key interfaces and integration points
        4. Usage examples with code snippets
        5. Important considerations and edge cases
        6. Performance implications if relevant

        Focus on providing information that developers need to effectively understand
        and work with this code.
        """,
    )


@register_tool(tags=["testing"])
def design_test_suite(action_context: ActionContext, feature_description: str) -> str:
    """
    Design a comprehensive test suite by consulting a senior QA engineer.
    This expert focuses on creating thorough test coverage with attention to edge cases.

    Args:
        feature_description: Description of the feature to test
    """
    return prompt_expert(
        action_context=action_context,
        description_of_expert="""
        You are a senior QA engineer with 12 years of experience in test design and automation.
        Your expertise includes:
        - Comprehensive test strategy development
        - Unit, integration, and end-to-end testing
        - Performance and stress testing
        - Security testing considerations
        - Test automation best practices

        You are particularly skilled at identifying edge cases and potential failure modes
        that others might miss. Your test suites are known for their thoroughness and
        their ability to catch issues early in the development cycle.
        """,
        prompt=f"""
        Please design a comprehensive test suite for the following feature:

        {feature_description}

        Your test design should cover:
        1. Unit tests for individual components
        2. Integration tests for component interactions
        3. End-to-end tests for critical user paths
        4. Performance test scenarios if relevant
        5. Edge cases and error conditions
        6. Test data requirements

        For each test category, provide:
        - Specific test scenarios
        - Expected outcomes
        - Important edge cases to consider
        - Potential testing challenges
        """,
    )


@register_tool(tags=["code_quality"])
def perform_code_review(action_context: ActionContext, code: str) -> str:
    """
    Review code and suggest improvements by consulting a senior software architect.
    This expert focuses on code quality, architecture, and best practices.

    Args:
        code: The code to review
    """
    return prompt_expert(
        action_context=action_context,
        description_of_expert="""
        You are a senior software architect with 20 years of experience in code review
        and software design. Your expertise includes:
        - Software architecture and design patterns
        - Code quality and maintainability
        - Performance optimization
        - Scalability considerations
        - Security best practices

        You have a talent for identifying subtle design issues and suggesting practical
        improvements that enhance code quality without over-engineering.
        """,
        prompt=f"""
        Please review the following code and provide detailed improvement suggestions:

        {code}

        Consider and address:
        1. Code organization and structure
        2. Potential design pattern applications
        3. Performance optimization opportunities
        4. Error handling completeness
        5. Edge case handling
        6. Maintainability concerns

        For each suggestion:
        - Explain the current issue
        - Provide the rationale for change
        - Suggest specific improvements
        - Note any trade-offs to consider
        """,
    )


@register_tool(tags=["communication"])
def write_feature_announcement(
    action_context: ActionContext, feature_details: str, audience: str
) -> str:
    """
    Write a feature announcement by consulting a product marketing expert.
    This expert focuses on clear communication of technical features to different audiences.

    Args:
        feature_details: Technical details of the feature
        audience: Target audience for the announcement (e.g., "technical", "business")
    """
    return prompt_expert(
        action_context=action_context,
        description_of_expert="""
        You are a senior product marketing manager with 12 years of experience in
        technical product communication. Your expertise includes:
        - Translating technical features into clear value propositions
        - Crafting compelling product narratives
        - Adapting messaging for different audience types
        - Building excitement while maintaining accuracy
        - Creating clear calls to action

        You excel at finding the perfect balance between technical accuracy and
        accessibility, ensuring your communications are both precise and engaging.
        """,
        prompt=f"""
        Please write a feature announcement for the following feature:

        {feature_details}

        This announcement is intended for a {audience} audience.

        Your announcement should include:
        1. A compelling introduction
        2. Clear explanation of the feature
        3. Key benefits and use cases
        4. Technical details (adapted to audience)
        5. Implementation requirements
        6. Next steps or call to action

        Ensure the tone and technical depth are appropriate for a {audience} audience.
        Focus on conveying both the value and the practical implications of this feature.
        """,
    )


### Here’s how dynamic expertise might be implemented
@register_tool()
def create_and_consult_expert(
    action_context: ActionContext, expertise_domain: str, problem_description: str
) -> str:
    """
    Dynamically create and consult an expert persona based on the specific domain and problem.

    Args:
        expertise_domain: The specific domain of expertise needed
        problem_description: Detailed description of the problem to be solved

    Returns:
        The expert's insights and recommendations
    """
    # Step 1: Dynamically generate a persona description
    persona_description_prompt = f"""
    Create a detailed description of an expert in {expertise_domain} who would be 
    ideally suited to address the following problem:

    {problem_description}

    Your description should include:
    - The expert's background and experience
    - Their specific areas of specialization within {expertise_domain}
    - Their approach to problem-solving
    - The unique perspective they bring to this type of challenge
    """

    generate_response = action_context.get("llm")
    persona_description = generate_response(
        Prompt(messages=[{"role": "user", "content": persona_description_prompt}])
    )

    # Step 2: Generate a specialized consultation prompt
    consultation_prompt_generator = f"""
    Create a detailed consultation prompt for an expert in {expertise_domain} 
    addressing the following problem:

    {problem_description}

    The prompt should guide the expert to provide comprehensive insights and
    actionable recommendations specific to this problem.
    """

    consultation_prompt = generate_response(
        Prompt(messages=[{"role": "user", "content": consultation_prompt_generator}])
    )

    # Step 3: Consult the dynamically created persona
    return prompt_expert(
        action_context=action_context,
        description_of_expert=persona_description,
        prompt=consultation_prompt,
    )


def develop_feature(action_context: ActionContext, feature_request: str) -> dict:
    """
    Process a feature request through a chain of expert personas.
    """
    # Step 1: Product expert defines requirements
    requirements = prompt_expert(
        action_context,
        "product manager expert",
        f"Convert this feature request into detailed requirements: {feature_request}",
    )

    # Step 2: Architecture expert designs the solution
    architecture = prompt_expert(
        action_context,
        "software architect expert",
        f"Design an architecture for these requirements: {requirements}",
    )

    # Step 3: Developer expert implements the code
    implementation = prompt_expert(
        action_context,
        "senior developer expert",
        f"Implement code for this architecture: {architecture}",
    )

    # Step 4: QA expert creates test cases
    tests = prompt_expert(
        action_context,
        "QA engineer expert",
        f"Create test cases for this implementation: {implementation}",
    )

    # Step 5: Documentation expert creates documentation
    documentation = prompt_expert(
        action_context,
        "technical writer expert",
        f"Document this implementation: {implementation}",
    )

    return {
        "requirements": requirements,
        "architecture": architecture,
        "implementation": implementation,
        "tests": tests,
        "documentation": documentation,
    }


@register_tool(tags=["invoice_processing", "categorization"])
def categorize_expenditure(action_context: ActionContext, description: str) -> str:
    """
    Categorize an invoice expenditure based on a short description.

    Args:
        description: A one-sentence summary of the expenditure.

    Returns:
        A category name from the predefined set of 20 categories.
    """
    categories = [
        "Office Supplies",
        "IT Equipment",
        "Software Licenses",
        "Consulting Services",
        "Travel Expenses",
        "Marketing",
        "Training & Development",
        "Facilities Maintenance",
        "Utilities",
        "Legal Services",
        "Insurance",
        "Medical Services",
        "Payroll",
        "Research & Development",
        "Manufacturing Supplies",
        "Construction",
        "Logistics",
        "Customer Support",
        "Security Services",
        "Miscellaneous",
    ]

    return prompt_expert(
        action_context=action_context,
        description_of_expert="A senior financial analyst with deep expertise in corporate spending categorization.",
        prompt=f"Given the following description: '{description}', classify the expense into one of these categories:\n{categories}",
    )


@register_tool(tags=["invoice_processing", "validation"])
def check_purchasing_rules(action_context: ActionContext, invoice_data: dict) -> dict:
    """
    Validate an invoice against company purchasing policies.

    Args:
        invoice_data: Extracted invoice details, including vendor, amount, and line items.

    Returns:
        A dictionary indicating whether the invoice is compliant, with explanations.
    """
    # Load the latest purchasing rules from disk
    rules_path = "config/purchasing_rules.txt"

    try:
        with open(rules_path, "r") as f:
            purchasing_rules = f.read()
    except FileNotFoundError:
        purchasing_rules = "No rules available. Assume all invoices are compliant."

    return prompt_expert(
        action_context=action_context,
        description_of_expert="A corporate procurement compliance officer with extensive knowledge of purchasing policies.",
        prompt=f"""
        Given this invoice data: {invoice_data}, check whether it complies with company purchasing rules.
        The latest purchasing rules are as follows:

        {purchasing_rules}

        Identify any violations or missing requirements. Respond with:
        - "compliant": true or false
        - "issues": A brief explanation of any problems found
        """,
    )


"""
1. All purchases over $5,000 require pre-approval.
2. IT equipment purchases must be from approved vendors.
3. Travel expenses must include a justification.
4. Consulting fees over $10,000 require an SOW (Statement of Work).
"""
{
    "invoice_number": "7890",
    "date": "2025-03-10",
    "vendor": "Tech Solutions Inc.",
    "total_amount": 6000,
    "line_items": [
        {"description": "High-end workstation", "quantity": 1, "total": 6000}
    ],
}

{
    "compliant": false,
    "issues": "This purchase exceeds $5,000 but lacks required pre-approval.",
}


@register_tool(tags=["invoice_processing", "validation"])
def check_purchasing_rules(action_context: ActionContext, invoice_data: dict) -> dict:
    """
    Validate an invoice against company purchasing policies, returning a structured response.

    Args:
        invoice_data: Extracted invoice details, including vendor, amount, and line items.

    Returns:
        A structured JSON response indicating whether the invoice is compliant and why.
    """
    rules_path = "config/purchasing_rules.txt"

    try:
        with open(rules_path, "r") as f:
            purchasing_rules = f.read()
    except FileNotFoundError:
        purchasing_rules = "No rules available. Assume all invoices are compliant."

    validation_schema = {
        "type": "object",
        "properties": {"compliant": {"type": "boolean"}, "issues": {"type": "string"}},
    }

    return prompt_llm_for_json(
        action_context=action_context,
        schema=validation_schema,
        prompt=f"""
        Given this invoice data: {invoice_data}, check whether it complies with company purchasing rules.
        The latest purchasing rules are as follows:

        {purchasing_rules}

        Respond with a JSON object containing:
        - `compliant`: true if the invoice follows all policies, false otherwise.
        - `issues`: A brief explanation of any violations or missing requirements.
        """,
    )


def create_invoice_agent():
    # Create action registry with invoice tools
    action_registry = PythonActionRegistry()

    # Define invoice processing goals
    goals = [
        Goal(
            name="Persona",
            description="You are an Invoice Processing Agent, specialized in handling invoices efficiently.",
        ),
        Goal(
            name="Process Invoices",
            description="""
            Your goal is to process invoices accurately. For each invoice:
            1. Extract key details such as vendor, amount, and line items.
            2. Generate a one-sentence summary of the expenditure.
            3. Categorize the expenditure using an expert.
            4. Validate the invoice against purchasing policies.
            5. Store the processed invoice with categorization and validation status.
            6. Return a summary of the invoice processing results.
            """,
        ),
    ]

    # Define agent environment
    environment = PythonEnvironment()

    return Agent(
        goals=goals,
        agent_language=AgentFunctionCallingActionLanguage(),
        action_registry=action_registry,
        generate_response=generate_response,
        environment=environment,
    )


invoice_text = """
    Invoice #4567
    Date: 2025-02-01
    Vendor: Tech Solutions Inc.
    Items: 
      - Laptop - $1,200
      - External Monitor - $300
    Total: $1,500
"""

# Create an agent instance
agent = create_invoice_agent()

# Process the invoice
response = agent.run(f"Process this invoice:\n\n{invoice_text}")

print(response)
