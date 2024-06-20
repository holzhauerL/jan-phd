import os                       # file operations
import json                     # JSON file operations
from enum import Enum           # enumeration
from datetime import datetime   # date and time operations
from tabulate import tabulate   # tabular output
import tqdm                     # progress bars
import tiktoken                 # token counting
import openai                   # OpenAI API    
from src.utils import Logger    # logging

class OpenAIModels(Enum):
    """
    Enumeration of OpenAI models. For an overview, see this page https://openai.com/pricing.
    Syntax: model_name = [model_id, input_price in US$ / 1M tokens, output_price in US$ / 1M tokens, type ("chat" or "embedding")]
    """
    GPT_3_5_TURBO_0125 = ["gpt-3.5-turbo-0125", 0.5, 1.5, "chat"]
    GPT_4_TURBO_2024_04_09 = ["gpt-4-turbo-2024-04-09", 10, 30, "chat"]
    GPT_4O_2024_05_13 = ["gpt-4o-2024-05-13", 5, 15, "chat"]
    TEXT_EMBEDDING_3_LARGE = ["text-embedding-3-large", 0.13, 0, "embedding"]

class CustomOpenAIClient():
    """
    Wrapper class to interact with the OpenAI API with custom functionalities such as cost estimation.
    """
    DEFAULT_MODEL = OpenAIModels.GPT_4O_2024_05_13
    MIN_OUTPUT_ESTIMATE = 0         # Minimum number of tokens to estimate the cost
    MAX_OUTPUT_ESTIMATE = 1000      # Maximum number of tokens to estimate the cost
    MIN_SAFETY = 0.5                # Minimum safety factor to multiply the estimated cost
    MAX_SAFETY = 2                  # Maximum safety factor to multiply the estimated cost
    MIN_TEMPERATURE = 0             # Minimum temperature to be used for the OpenAI API
    MAX_TEMPERATURE = 2             # Maximum temperature to be used for the OpenAI API
    MIN_COST_LIMIT = 0              # Minimum cost threshold in USD for a single call
    MAX_COST_LIMIT = 20             # Maximum cost threshold in USD for a single call
    TOKENS_PER_USD = 1000000        # Conversion factor to account for how prices are given in USD per 1M tokens
    DEFAULT_TOKENS_PER_MESSAGE = 2  # Default number of tokens per message for cost estimation

    def __init__(self, model=DEFAULT_MODEL, estimation_mode=True, max_tokens=1000, output_estimate=30, safety=1.1, temperature=0.7, cost_limit_single_call=0.1, log=True, verbose=True):
        """
        Initializes the OpenAI client and creates a log file.

        :param model: The model to be used for the OpenAI API. Defaults to 'gpt-3.5-turbo-0125'.
        :param estimation_mode: Whether to just estimate the cost of the API call or actually call the API, which results in costs. Defaults to True.
        :param max_tokens: The maximum number of tokens to be used for the OpenAI API. Defaults to 1000.
        :param output_estimate: The number of tokens to estimate the cost. Defaults to 30.
        :param safety: The safety factor to multiply the estimated cost. Defaults to 1.1.
        :param temperature: The temperature to be used for the OpenAI API. Defaults to 0.7.
        :param cost_warning_single_call: The cost threshold in USD to ask the user for confirmation before calling the API. Defaults to $0.1. 
        :param log: Whether to create a log file. Defaults to True.
        :param verbose: Whether to print the outputs. If set to False, no outputs apart from the initialization are printed, also not in the log file. Defaults to True.
        """
        self.model = self._validate_model(model)
        self.model_name = self.model[0]
        self.input_price = self.model[1]
        self.output_price = self.model[2]
        self.type = self.model[3]


        self.estimation_mode = estimation_mode
        self.max_tokens = self._validate_max_tokens(max_tokens)
        self.output_estimate = self._validate_output_estimate(output_estimate)
        self.safety = self._validate_safety(safety)
        self.temperature = self._validate_temperature(temperature)
        self.cost_limit_single_call = self._validate_cost_limit(cost_limit_single_call)

        self.resources = {
            'Costs': {'Input': 0.0, 'Output': 0.0, 'Total': 0.0},
            'Estimated costs': {'Input': 0.0, 'Output': 0.0, 'Total': 0.0},
            'Tokens': {'Input': 0, 'Output': 0},
            'Calls' : 0
        }

        self.verbose = verbose
        self.log = log
        self.logger = Logger(log=log)
        self._print_initialization()

    def _print_initialization(self):
        @self.logger.log_print
        def print_init():
            print(f"OpenAIClient created at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with the following parameters:")
            print(f"Model: {self.model_name}")
            print(f"Estimation mode: {self.estimation_mode}")
            print(f"Output estimate: {self.output_estimate}")
            print(f"Safety factor: {self.safety}")
            print(f"Temperature: {self.temperature}")
            print(f"Cost limit single call: US$ {self.cost_limit_single_call}")
        return print_init()

    def _validate_model(self, model):
        if not isinstance(model, OpenAIModels):
            print(f"Warning: Model {model} not found. Using default model {self.DEFAULT_MODEL.value}.")
            return self.DEFAULT_MODEL.value
        return model.value
    
    def _validate_max_tokens(self, max_tokens):
        if not max_tokens > 0:
            raise ValueError(f"The maximum number of tokens max_tokens must be at least 1. The current value is {max_tokens}.")
        if not isinstance(max_tokens, int):
            raise ValueError(f"The maximum number of tokens max_tokens must be an integer. The current type is {type(max_tokens)}.")
        return max_tokens

    def _validate_output_estimate(self, output_estimate):
        if not output_estimate >= self.MIN_OUTPUT_ESTIMATE:
            raise ValueError(f"The output estimate must be at least {self.MIN_OUTPUT_ESTIMATE}. The current value is {output_estimate}.")
        return output_estimate

    def _validate_safety(self, safety):
        if not (self.MIN_SAFETY <= safety <= self.MAX_SAFETY):
            raise ValueError(f"The safety factor must be between {self.MIN_SAFETY} and {self.MAX_SAFETY}. The current value is {safety}.")
        return safety

    def _validate_temperature(self, temperature):
        if not (self.MIN_TEMPERATURE <= temperature <= self.MAX_TEMPERATURE):
            raise ValueError(f"The temperature must be between {self.MIN_TEMPERATURE} and {self.MAX_TEMPERATURE}. The current value is {temperature}.")
        return temperature

    def _validate_cost_limit(self, cost_limit_single_call):
        if not (self.MIN_COST_LIMIT <= cost_limit_single_call <= self.MAX_COST_LIMIT):
            raise ValueError(f"The cost limit for a single call must be between {self.MIN_COST_LIMIT} and {self.MAX_COST_LIMIT}. The current value is {cost_limit_single_call}.")
        return cost_limit_single_call

    def num_tokens_from_messages(self, messages, model=None):
        """Return the number of tokens used by a list of messages. Also works, if messages is a single string.
        Taken from: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken. 

        :param messages: List of dictionaries containing the messages for the OpenAI API.
        :param model: The model to be used for the encoding. Defaults to "gpt-4-0613".
        :return: The number of tokens used by the messages.
        """
        messages = self._check_messages(messages)

        if model is None:
            model = self.model_name if self.model is not None else "gpt-4-0613"

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            # print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            # print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            # raise NotImplementedError(
            #     f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            # )
            tokens_per_message = self.DEFAULT_TOKENS_PER_MESSAGE
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
    
    def _calculate_costs(self, input_tokens, output_tokens):
        """
        Calculates the costs based on the number of tokens used.

        :param input_tokens: The number of tokens used for the input.
        :param output_tokens: The number of tokens used for the output.
        :return: The costs for the input, output, and total costs.
        """
        input_cost = self.input_price * input_tokens * self.safety / self.TOKENS_PER_USD
        output_cost = self.output_price * output_tokens * self.safety / self.TOKENS_PER_USD
        total_costs = input_cost + output_cost

        return input_cost, output_cost, total_costs

    def _check_messages(self, messages):
        """
        Creates a message with the given messages, if they are only a string.

        :param messages: The messages to be checked.
        :return: A list of messages in the format expected by the OpenAI API.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if isinstance(messages, list):
            return messages
        else:
            raise ValueError("Messages must be either a list of dictionaries or a string.")
    
    def _concat_messages(self, messages):
        """
        Formats the messages into a single string for better readability.
        
        :param messages: The messages to be formatted.
        :return: The formatted messages.
        """
        messages_checked = self._check_messages(messages)
        # Concatenate the messages into a single organized string 
        concat_message = " ".join([message["content"] for message in messages_checked])
        # Add quotes around the message
        concat_message = f'"{concat_message}"'
        # If it is too long, cut it off, add "[<number_of_characters_between>]" and then add the last 50 characters
        if len(concat_message) > 200:
            concat_message = f'{concat_message[:100]} ... [{len(concat_message) - 150} more characters] ... {concat_message[-50:]}'
        return concat_message
    
    def estimate_resources(self, messages):
            """
            Estimates the cost of the API call based on the number of tokens used.

            :param messages: The messages to be sent to the OpenAI API.
            :return: The estimated total costs of the API call.
            """
            @self.logger.log_print
            def estimate_resources_():
                messages_checked = self._check_messages(messages)
                input_tokens = self.num_tokens_from_messages(messages_checked, model=self.model_name)
                output_tokens = self.output_estimate
                input_cost, output_cost, total_costs = self._calculate_costs(input_tokens, output_tokens)

                concat_message = self._concat_messages(messages_checked)

                if self.verbose:
                    # Print the estimated costs in tabular format
                    print("\nESTIMATED RESOURCES - SINGLE API CALL")
                    print(f"Text: {concat_message}")
                    single_call_data = [
                        ["Input tokens", input_tokens],
                        ["Output tokens", output_tokens],
                        ["Input cost [$]", input_cost],
                        ["Output cost [$]", output_cost],
                        ["Total cost [$]", total_costs]
                    ]
                    print(tabulate(single_call_data, headers=["Name", "Estimation"]))

                return total_costs
            
            return estimate_resources_()

    def calculate_resources(self, messages, input_tokens, output_tokens):
            """
            Calculates the resources used by the API call and updates the resources dictionary.

            :param messages: The messages to be sent to the OpenAI API.
            :param input_tokens: The number of tokens used for the input.
            :param output_tokens: The number of tokens used for the output.
            """
            @self.logger.log_print
            def calculate_resources_():
                messages_checked = self._check_messages(messages)

                # Estimate costs for quality control
                input_tokens_estimate = self.num_tokens_from_messages(messages_checked, model=self.model_name)
                output_tokens_estimate = self.output_estimate
                input_cost_estimate, output_cost_estimate, total_costs_estimate = self._calculate_costs(input_tokens_estimate, output_tokens_estimate)

                # Calculate actual costs
                input_cost, output_cost, total_costs = self._calculate_costs(input_tokens, output_tokens)

                # Update the resources dictionary
                self.resources['Estimated costs']['Input'] += input_cost_estimate
                self.resources['Estimated costs']['Output'] += output_cost_estimate
                self.resources['Estimated costs']['Total'] += total_costs_estimate
                self.resources['Costs']['Input'] += input_cost
                self.resources['Costs']['Output'] += output_cost
                self.resources['Costs']['Total'] += total_costs
                self.resources['Tokens']['Input'] += input_tokens
                self.resources['Tokens']['Output'] += output_tokens
                self.resources['Calls'] += 1

                # Concatenate the messages into a single organized string
                concat_message = self._concat_messages(messages_checked)

                if self.verbose:
                    # Print the actual costs in tabular format
                    print("\nRESOURCES - SINGLE API CALL")
                    print(f"Text: {concat_message}")
                    single_call_data = [
                        ["Input tokens", input_tokens, input_tokens_estimate, self._quotient(input_tokens, input_tokens_estimate)],
                        ["Output tokens", output_tokens, output_tokens_estimate, self._quotient(output_tokens, output_tokens_estimate)],
                        ["Input cost [$]", input_cost, input_cost_estimate, self._quotient(input_cost, input_cost_estimate)],
                        ["Output cost [$]", output_cost, output_cost_estimate, self._quotient(output_cost, output_cost_estimate)],
                        ["Total cost [$]", total_costs, total_costs_estimate, self._quotient(total_costs, total_costs_estimate)]
                    ]
                    print(tabulate(single_call_data, headers=["Name", "Value (V)", "Estimation (E)", "V/E"]))

                    # Print the aggregated costs in tabular format
                    print("\nRESOURCES - AGGREGATION SINCE INSTANCE WAS CREATED")
                    aggregation_data = [
                        ["Input costs [$]", self.resources['Costs']['Input'], self.resources['Estimated costs']['Input'], self._quotient(self.resources['Costs']['Input'], self.resources['Estimated costs']['Input'])],
                        ["Output costs [$]", self.resources['Costs']['Output'], self.resources['Estimated costs']['Output'], self._quotient(self.resources['Costs']['Output'], self.resources['Estimated costs']['Output'])],
                        ["Total costs [$]", self.resources['Costs']['Total'], self.resources['Estimated costs']['Total'], self._quotient(self.resources['Costs']['Total'], self.resources['Estimated costs']['Total'])],
                        ["Calls", self.resources['Calls'], "-", "-"],
                        ["Cost per call [$]", self._quotient(self.resources['Costs']['Total'], self.resources['Calls']), "-", "-"],
                    ]
                    print(tabulate(aggregation_data, headers=["Name", "Value (V)", "Estimation (E)", "V/E"]))

            return calculate_resources_()
    
    def _quotient(self, a, b):
        """
        Calculates the quotient of two numbers without division by zero. Only return up to two decimal places.

        :param a: The numerator.
        :param b: The denominator.
        :return: The quotient of the two numbers.
        """
        return round(a / b, 2) if b != 0 else 0
    
    def single_api_call(self, input, model=None, temperature=None, max_tokens=None):
        """
        Calls the OpenAI API with the given input and returns the response.

        :param input: The input to be sent to the OpenAI API. For chat completions, the input must be a list of dictionaries or a string. For text embeddings, the input must be a string.
        :param model: The model to be used for the OpenAI API. Defaults to None.
        :param temperature: The temperature to be used for the OpenAI API. Only relevant for chat completions. Defaults to None. 
        :param max_tokens: The maximum number of tokens to be used for the OpenAI API. Only relevant for chat completions. Defaults to None.
        :return: The response from the OpenAI API. For chat completions, this is the most probable response as a string. For text embeddings, this is the embedding vector. 
        """
        if model is None:
            model = self.model
        model_name = model[0]

        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Check the format of the input
        messages_checked = self._check_messages(input)
        # Estimate the cost of the API call
        total_costs_estimate = self.estimate_resources(messages_checked)

        # Round the total costs estimate to two four places
        total_costs_estimate = round(total_costs_estimate, 4)

        # Abort if the total costs are greater than the cost limit for a single call
        if total_costs_estimate > self.cost_limit_single_call:
            print(f'\nAPI call aborted.\n\n Input: "{self._concat_messages(input)}"\n\nCost estimate of ${total_costs_estimate} exceeds the defined limit of a single call of ${self.cost_limit_single_call}.\n')
            return None
        
        # Call the OpenAI API, but only if the estimation mode is set to False
        if not self.estimation_mode:
            client = openai.OpenAI()

            if self.type == "chat":
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages_checked,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                response = completion.choices[0].message.content
                input_tokens = completion.usage.prompt_tokens
                output_tokens = completion.usage.completion_tokens
            elif self.type == "embedding" and isinstance(input, str):
                embedding = client.embeddings.create(
                    model=model_name,
                    input = [input]
                )
                response = embedding.data[0].embedding
                input_tokens = embedding.usage.total_tokens
                output_tokens = 0
            else:
                raise ValueError('\nType must be either "chat" or "embedding". If type is "embedding", messages must be a string.\n')
            self.calculate_resources(messages_checked, input_tokens, output_tokens)
        else:
            response = None

        return response
    
    def load_json(self, folder_path, file_path):
        """
        Loads the JSON file from the given path.

        :param folder_path: The path to the folder where the JSON file is saved.
        :param file_path: The path to the JSON file.
        :return: The JSON file.
        """
        # Check if the file path is a JSON file
        if not file_path.endswith('.json'):
            raise ValueError("The file path must be a JSON file.")
        # If the file does not exist, it will be created
        if not os.path.exists(file_path):

            # If the folder does not exist, it will be created
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            dict = {"_": None}
            # Create the JSON file and save the dict in it
            with open(file_path, 'w') as json_file:
                json.dump(dict, json_file, indent=4)

        # Load the embeddings from the file
        with open(file_path, 'r') as f:
            dict = json.load(f)

        # Close the file
        f.close()

        return dict
    
    def batch_api_call(self, input, folder_path_responses, file_path_responses, cost_limit_batch, update_counter=100):
        """
        Retrieves the responses for the given input from the OpenAI API if they don't exist yet and saves them to a file.

        :param input: The input for which the responses should be retrieved. Can be a list of strings or a dictionary with strings as values and keys to identify the existing responses.
        :param folder_path_responses: The path to the folder where the responses file should be saved.
        :param file_path_responses: The path to the responses file.
        :param cost_limit_batch: The cost limit for the batch of API calls.
        :param update_counter: The number of API calls after which the costs are printed. Defaults to 100.

        :return: The responses and a list of input items for which the API call failed.
        """
        @self.logger.log_print
        def batch_api_call_():
            
            # Load the embeddings from the file
            database = self.load_json(folder_path_responses, file_path_responses)

            # Create a set of the existing embeddings
            existing_responses = set(database.keys())

            # If the input is a dictionary, use the keys as the query
            if isinstance(input, dict):
                input_is_dictionary = True
                query = set(input.keys())
            elif isinstance(input, list):
                input_is_dictionary = False
                query = set(input)
            else:
                raise ValueError("Input must be either a list or a dictionary.")

            # Create a set of the sentences that need to be embedded, to avoid unnecessary API calls
            query_to_call = query - existing_responses

            # Avoid overflow of print statements and allow tqdm to work
            self.logger.log = False
            old_verbose = self.verbose
            self.verbose = False

            # Estimate the resources for the API calls
            total_costs_estimate = 0
            print("\nEstimating resources for the API calls...\n")

            for q in query_to_call:
                if input_is_dictionary:
                    total_costs_estimate += self.estimate_resources(input[q])
                else:
                    total_costs_estimate += self.estimate_resources(q)

            # Round the total costs estimate to two four places
            total_costs_estimate = round(total_costs_estimate, 4)

            # Abort if the total costs are greater than the cost limit for a single call
            if total_costs_estimate > cost_limit_batch:
                print(f'\nBatch API call aborted.\n\nCost estimate of ${total_costs_estimate} exceeds the defined limit for the batch of ${cost_limit_batch}.\n')
                return None
            else:
                print(f'\nCost estimate of ${total_costs_estimate} is within the defined limit for the batch of ${cost_limit_batch}.\n')
            
            exceptions = []

            if not self.estimation_mode:

                num_calls = len(query_to_call)
                counter = 1
                exception_counter = 0

                # Apply tqdm
                for q in tqdm.tqdm(query_to_call, desc="Calling OpenAI API", unit="calls"):
                    # For every update_counter-th query item and for the last one, print the costs
                    if counter % update_counter == 0 or counter == num_calls:
                        self.logger.log = self.log
                        self.verbose = True
                    # Try to call the API
                    try:
                        # Call the API
                        if input_is_dictionary:
                            response = self.single_api_call(input[q])
                        else:   
                            response = self.single_api_call(q)

                        # Save the response to the dictionary
                        database[q] = response

                    except Exception as e:
                        print(f"\nError: {e} for query item: {self._concat_messages(q)}\n")
                        exceptions.append((e, q))
                        exception_counter += 1
                    finally:
                        counter += 1
                        self.logger.log = False
                        self.verbose = False
                
                # Print the number of exceptions out of the total number of API calls
                print(f"\nExceptions: {exception_counter} out of {num_calls} API calls.\n")

                # Update the database
                self.save_json(database, file_path_responses)

            # Reset the verbose and log attributes
            self.logger.log = self.log
            self.verbose = old_verbose

            return database, exceptions
        return batch_api_call_()
    
    def save_json(self, dict, file_path):
        """
        Saves the dictionary to the JSON file.

        :param dict: The dictionary to be saved.
        :param file_path: The path to the JSON file.
        """
        # Save the embeddings to the file
        with open(file_path, 'w') as f:
            json.dump(dict, f, indent=4)
        # Close the file
        f.close()