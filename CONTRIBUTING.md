This guide provides information on how to contribute code in a consistent and maintainable way. For information on getting started with the project, please refer to the [onboarding guide](https://azaho.github.io/bfm/getting-started).

## Workflow

1. **Create a branch** for your work:
    ```bash
    git checkout -b feature/your-feature-name
    ```
2. **Make your changes** and commit them with clear messages.

3. **Push your branch** and open a pull request (PR).

4. **Request a review** from at least one other team member.

## Documentation

We use **[MkDocs](https://www.mkdocs.org/)** and **[mkdocstrings](https://mkdocstrings.github.io/)** for documentation.

This means **docstrings** in the code are used to generate the documentation automatically. Please write **Google-style docstrings** for all functions and classes:

```python
def add_numbers(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The sum of `a` and `b`.
    """
    return a + b
```

For short functions, you can use a one-line description.

## Code Style & Linting

We use **[Ruff](https://ruff.rs/)** for linting and formatting.

- Check for issues:  
  ```bash
  ruff check .
    ```
- Automatically fix issues:
    ```bash
    ruff check . --fix
    ```
- Format code:
    ```bash
    ruff format .
    ```

## Tests

- Every new feature or bugfix should include a small test.

- Run tests with: 
    ```bash
    pytest
    ```

## Guidelines

- Write clear, concise, and maintainable code.

- Keep commits reasonably small and clear. We recommend using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

- Ask questions if you're unsure about anything. We're happy to help!