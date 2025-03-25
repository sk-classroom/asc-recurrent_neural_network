# %% Import
import numpy as np

# %% Define utility functions

def longShortTermMemory(input, hidden_state, cell_state):
    """
    LSTM implementation using a stateless approach for processing sequences.
    Processes input through forget, input, and output gates.

    Args:
        input: Current input to the LSTM cell
            Single numeric value to process
        hidden_state: Hidden state from previous step
            Array [min_value, padding] tracking minimum value
        cell_state: Internal cell state from previous step
            Array tracking running [sum, product, min] values

    Returns:
        tuple:
            - hidden_state: Updated [min_value, padding] for next step
            - cell_state: Updated running [sum, product, min] values
            - output: Current [sum, product, min] outputs
    """
    # Forget gate: Determines what information to discard from the cell state
    # Returns values between 0 (forget) and 1 (keep) for each element
    forget_factor = forget_gate(input, hidden_state)
    assert all(
        0 <= x <= 1 for x in forget_factor
    ), "Forget gate outputs must be between 0 and 1"

    # Input gate: Decides what new information to store in the cell state
    # Processes current input and previous hidden state to generate new values
    added_values = input_gate(input, hidden_state)

    # Update cell state:
    # 1. Multiply old cell state by forget factor (forgetting irrelevant info)
    # 2. Add new values from input gate (adding new information)
    cell_state = cell_state * forget_factor + added_values

    assert all(
        -60 <= x <= 60 for x in cell_state
    ), "Cell state must be between -60 and 60"


    # Output gate:
    # 1. Filters the cell state to determine what to output
    # 2. Updates the hidden state for the next time step
    output, hidden_state = output_gate(input, hidden_state, cell_state)

    return (
        hidden_state,  # [min_value, padding] for next step
        cell_state,  # Updated running [sum, product, min]
        output,  # Current [sum, product, min] outputs
    )

class MemoryGame:
    def __init__(
        self, seq_len=10, stm_len=3, integer_set=list(range(10)), level=1
    ):
        self.seq_len = seq_len
        self.stm_len = stm_len
        self.integer_set = list(set(integer_set))
        self.current_index = None
        self.sequence = None
        self.stm = None
        self.question_index = None
        self.result = "none"
        self.level = 1
        self.replacement = True

        self.questions = []
        self.question_pool = [
            # Level 1 -------------------------------------------------------
            {
                "question": "What was the sum of all numbers?",
                "func": self.get_sum,
                "level": 1,
            },
            {
                "question": "What was the sum of even numbers ",
                "func": lambda: np.sum(
                    [s for s in self.sequence if s % 2 == 0]
                ),
                "level": 1,
            },
            {
                "question": "How many evens followed the final odd? (Total length if no odds)",
                "func": lambda: len(self.sequence)
                - np.where(np.array(self.sequence) % 2 == 1)[0][-1]
                - 1
                if any(np.array(self.sequence) % 2 == 1)
                else len(self.sequence),
                "level": 1,
            },
            # Level 2 -------------------------------------------------------
            {
                "question": "How many odd numbers in a row at their last appearance?",
                "func": self.get_len_last_odd_in_row,
                "level": 2,
            },
            {
                "question": "What was the most frequent number in the sequence? If ties, answer the smallest one.",
                "func": self.get_mode,
                "level": 2,
            },
            # Level 2 -------------------------------------------------------
            {
                "question": "What was the smallest number in the first four numbers?",
                "func": lambda: np.min(self.sequence[:4]),
                "level": 3,
            },
            {
                "question": "What was the largest number in the last four numbers?",
                "func": lambda: np.max(self.sequence[-4:]),
                "level": 3,
            },
            # Level 4 -------------------------------------------------------
            {
                "question": "What was the median?",
                "func": self.get_median,
                "level": 4,
            },
        ]
        self.hints = [
            "",
            "To solve this problem, consider how you would track and count the frequency of each number given the numbers and sequence length.",
            "For LSTM implementation: Use 'hidden_state[2]' as a control variable to manage LSTM behavior. When hidden_state[2] = 0, the LSTM will store values in memory. When hidden_state[2] > 0, the LSTM will perform calculations to summarize stored values. When hidden_state[2] < 0, the LSTM will modify its compression operation. You can define how 'hidden_state[2]' transitions between states based on the current cell and hidden states in the output gate.",
            "For LSTM implementation: Note that the median is the fourth smallest/largest number in the sequence. You'll need to utilize hidden_state[2] as a control variable in your LSTM logic.",
        ]

        self.reset()

    def set_level(self, level):
        if level == 1:  # Easy
            self.reset(
                seq_len=7,
                stm_len=3,
                integer_set={1, 2, 4, 8},
                level=level,
                replacement=True,
            )
        if level == 2:  # Medium (modified)
            self.reset(
                seq_len=7,
                stm_len=3,
                integer_set={1, 2, 3},
                level=level,
                replacement=True,
            )
        if level == 3:  # Medium (modified)
            self.reset(
                seq_len=7,
                stm_len=3,
                integer_set={1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                level=level,
                replacement=False,
            )
        if level == 4:  # Master
            self.reset(
                seq_len=7,
                stm_len=3,
                integer_set={1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                level=level,
                replacement=False,
            )

    def update_stm(self, number):
        self.stm.append(number)
        if len(self.stm) > self.stm_len:
            self.stm.pop(0)

    def get_current_value(self):
        return self.current_index

    def reset(
        self,
        seq_len=None,
        stm_len=None,
        integer_set=None,
        level=None,
        replacement=None,
    ):
        self.seq_len = seq_len if seq_len is not None else self.seq_len
        self.stm_len = stm_len if stm_len is not None else self.stm_len
        self.replacement = (
            replacement if replacement is not None else self.replacement
        )
        self.integer_set = (
            list(set(integer_set))
            if integer_set is not None
            else self.integer_set
        )
        self.level = level if level is not None else self.level

        self.sequence = np.random.choice(
            self.integer_set, size=self.seq_len, replace=self.replacement
        )
        self.stm = []
        self.current_index = 0

        self.questions = [
            q for q in self.question_pool if q["level"] == self.level
        ]
        self.question_index = np.random.randint(0, len(self.questions))
        self.result = "none"

    def update_current_index(self):
        if self.current_index < self.seq_len - 1:
            self.current_index += 1

    def get_sub_message(self):
        question = self.questions[self.question_index]["question"]
        answer = self.questions[self.question_index]["func"]()
        answer = float(answer)

        if self.result == "correct":
            return f"""
            Correct! 🥳
            """

        elif self.result == "incorrect":
            return f"""
            Incorrect 🤔. The answer was {answer:.1f}
            """

        return ""

    def get_message(self):
        progress = self.current_index + 1
        numbers_left = self.seq_len - progress - self.stm_len + 1
        message = ""

        question = self.questions[self.question_index]["question"]
        answer = self.questions[self.question_index]["func"]()
        answer = float(answer)

        if self.current_index < self.seq_len - self.stm_len:
            if self.current_index == 0:
                message = "Remember the numbers as they appear! "
            else:
                message = f"{numbers_left} more number{'s' if numbers_left > 1 else ''} to go "
        else:
            message = self.questions[self.question_index]["question"]
        return message

    def get_message_type(self):
        if self.current_index < self.seq_len - self.stm_len:
            return "neutral"
        elif self.result == "correct":
            return "success"
        elif self.result == "incorrect":
            return "danger"
        else:
            return "info"

    def eval(self, answer):
        try:
            answer = float(answer)
        except ValueError:
            return

        correct_answer = self.questions[self.question_index]["func"]()
        if np.abs(answer - correct_answer) < 0.1:
            self.result = "correct"
        else:
            self.result = "incorrect"

    def get_smallest(self):
        return np.min(self.sequence)

    def get_largest(self):
        return np.max(self.sequence)

    def get_average(self):
        return np.mean(self.sequence)

    def get_product(self):
        return np.prod(self.sequence)

    def get_sum(self):
        return np.sum(self.sequence)

    def get_sum_squared(self):
        return np.sum(self.sequence**2)

    def get_second_smallest(self):
        return np.sort(self.sequence)[1]

    def get_median(self):
        return np.median(self.sequence)

    def get_num_numbers_after_last_odd(self):
        cnt = 0
        for i in range(len(self.sequence)):
            if self.sequence[-(i + 1)] % 2 == 0:
                cnt += 1
            break
        return cnt

    def get_len_last_odd_in_row(self):
        cnt = 0
        for i in range(len(self.sequence)):
            if self.sequence[i] % 2 == 0:
                cnt = 0
            else:
                cnt += 1
        return cnt

    def get_last_four_pos_number(self):
        return selq.sequence[-4]

    def get_mode(self):
        labs, freq = np.unique(self.sequence, return_counts=True)
        indices = np.where(np.max(freq) == freq)[0]
        return labs[indices[0]]

num_tests = 100
hidden_state_size = 3
cell_state_size = 2

game = MemoryGame()
game.set_level(4)

def run_lstm(sequence, hidden_state_size, cell_state_size):
    """
    Run LSTM over a sequence of inputs.

    Args:
        sequence: Input sequence to process
        hidden_state_size: Size of hidden state array [min_value, padding]
        cell_state_size: Size of cell state array [sum, product, min]

    Returns:
        output: Final [sum, product, min] outputs after processing sequence
    """
    hidden_state = np.zeros(
        hidden_state_size
    )  # Initialize [min_value, padding]
    cell_state = np.zeros(cell_state_size)  # Initialize [sum, product, min]

    for s in sequence:
        hidden_state, cell_state, output = longShortTermMemory(
            s, hidden_state, cell_state
        )
    return output


# Run tests and compute accuracy
n_questions = len(game.questions)
n_correct = np.zeros(n_questions, dtype=float)
for i in range(num_tests):
    game.reset()
    output = run_lstm(game.sequence, hidden_state_size, cell_state_size)
    answers = np.array([q["func"]() for q in game.questions])

    output = np.array(output)[: len(answers)]
    # Compute the accuracy for each run
    n_correct += np.isclose(answers, output, atol=1e-1)

n_correct /= num_tests
final_score = 100 * np.min(n_correct)

print(f"Final Score: {final_score:.2f}%")
assert final_score > 90.0, "Final score is less than 90.0%"
# %% Test -----------
