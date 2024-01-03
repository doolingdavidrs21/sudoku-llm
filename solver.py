import pandas as pd
import plotly.express as px
#import altair as alt
import re
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import math
import plotly.io as pio
#import altair as alt
import pickle
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import numpy as np
from random import shuffle
import copy
from dotenv import load_dotenv
import os
import ast
import openai
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS # in memory

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA


#st.set_page_config(layout="wide")

embeddings = OpenAIEmbeddings()


class SudokuGenerator9X9:
	"""generates and solves Sudoku puzzles using a backtracking algorithm"""
	def __init__(self,grid=None):
		self.counter = 0
		#path is for the matplotlib animation
		self.path = []
		#if a grid/puzzle is passed in, make a copy and solve it
		if grid:
			if len(grid[0]) == 9 and len(grid) == 9:
				self.grid = grid
				self.original = copy.deepcopy(grid)
				self.solve_input_sudoku()
			else:
				print("input needs to be a 9x9 matrix")
		else:
			#if no puzzle is passed, generate one
			self.grid = [[0 for i in range(9)] for j in range(9)]
			self.generate_puzzle()
			self.original = copy.deepcopy(self.grid)
		
		
	def solve_input_sudoku(self):
		"""solves a puzzle"""
		self.generate_solution(self.grid)
		return

	def generate_puzzle(self):
		"""generates a new puzzle and solves it"""
		self.generate_solution(self.grid)
		#self.print_grid('full solution')
		self.remove_numbers_from_grid()
	#	self.print_grid('with removed numbers')
		return

	def print_grid(self, grid_name=None):
		if grid_name:
			print(grid_name)
		for row in self.grid:
			print(row)
		return

	def test_sudoku(self,grid):
		"""tests each square to make sure it is a valid puzzle"""
		for row in range(9):
			for col in range(9):
				num = grid[row][col]
				#remove number from grid to test if it's valid
				grid[row][col] = 0
				if not self.valid_location(grid,row,col,num):
					return False
				else:
					#put number back in grid
					grid[row][col] = num
		return True

	def num_used_in_row(self,grid,row,number):
		"""returns True if the number has been used in that row"""
		if number in grid[row]:
			return True
		return False

	def num_used_in_column(self,grid,col,number):
		"""returns True if the number has been used in that column"""
		for i in range(9):
			if grid[i][col] == number:
				return True
		return False

	def num_used_in_subgrid(self,grid,row,col,number):
		"""returns True if the number has been used in that subgrid/box"""
		sub_row = (row // 3) * 3
		sub_col = (col // 3)  * 3
		for i in range(sub_row, (sub_row + 3)): 
			for j in range(sub_col, (sub_col + 3)): 
				if grid[i][j] == number: 
					return True
		return False

	def valid_location(self,grid,row,col,number):
		"""return False if the number has been used in the row, column or subgrid"""
		if self.num_used_in_row(grid, row,number):
			return False
		elif self.num_used_in_column(grid,col,number):
			return False
		elif self.num_used_in_subgrid(grid,row,col,number):
			return False
		return True

	def find_empty_square(self,grid):
		"""return the next empty square coordinates in the grid"""
		for i in range(9):
			for j in range(9):
				if grid[i][j] == 0:
					return (i,j)
		return

	def solve_puzzle(self, grid):
		"""solve the sudoku puzzle with backtracking"""
		for i in range(0,81):
			row=i//9
			col=i%9
			#find next empty cell
			if grid[row][col]==0:
				for number in range(1,10):
					#check that the number hasn't been used in the row/col/subgrid
					if self.valid_location(grid,row,col,number):
						grid[row][col]=number
						if not self.find_empty_square(grid):
							self.counter+=1
							break
						else:
							if self.solve_puzzle(grid):
								return True
				break
		grid[row][col]=0  
		return False

	def generate_solution(self, grid):
		"""generates a full solution with backtracking"""
		number_list = [1,2,3,4,5,6,7,8,9]
		for i in range(0,81):
			row=i//9
			col=i%9
			#find next empty cell
			if grid[row][col]==0:
				shuffle(number_list)      
				for number in number_list:
					if self.valid_location(grid,row,col,number):
						self.path.append((number,row,col))
						grid[row][col]=number
						if not self.find_empty_square(grid):
							return True
						else:
							if self.generate_solution(grid):
								#if the grid is full
								return True
				break
		grid[row][col]=0  
		return False

	def get_non_empty_squares(self,grid):
		"""returns a shuffled list of non-empty squares in the puzzle"""
		non_empty_squares = []
		for i in range(len(grid)):
			for j in range(len(grid)):
				if grid[i][j] != 0:
					non_empty_squares.append((i,j))
		shuffle(non_empty_squares)
		return non_empty_squares

	def remove_numbers_from_grid(self):
		"""remove numbers from the grid to create the puzzle"""
		#get all non-empty squares from the grid
		non_empty_squares = self.get_non_empty_squares(self.grid)
		non_empty_squares_count = len(non_empty_squares)
		rounds = 3
		while rounds > 0 and non_empty_squares_count >= 17:
			#there should be at least 17 clues
			row,col = non_empty_squares.pop()
			non_empty_squares_count -= 1
			#might need to put the square value back if there is more than one solution
			removed_square = self.grid[row][col]
			self.grid[row][col]=0
			#make a copy of the grid to solve
			grid_copy = copy.deepcopy(self.grid)
			#initialize solutions counter to zero
			self.counter=0      
			self.solve_puzzle(grid_copy)   
			#if there is more than one solution, put the last removed cell back into the grid
			if self.counter!=1:
				self.grid[row][col]=removed_square
				non_empty_squares_count += 1
				rounds -=1
		return

class SudokuGenerator4X4:
	"""generates and solves Sudoku puzzles using a backtracking algorithm"""
	def __init__(self,grid=None):
		self.counter = 0
		#path is for the matplotlib animation
		self.path = []
		#if a grid/puzzle is passed in, make a copy and solve it
		if grid:
			if len(grid[0]) == 4 and len(grid) == 4:
				self.grid = grid
				self.original = copy.deepcopy(grid)
				self.solve_input_sudoku()
			else:
				print("input needs to be a 4x4 matrix")
		else:
			#if no puzzle is passed, generate one
			self.grid = [[0 for i in range(4)] for j in range(4)]
			self.generate_puzzle()
			self.original = copy.deepcopy(self.grid)
		
		
	def solve_input_sudoku(self):
		"""solves a puzzle"""
		self.generate_solution(self.grid)
		return

	def generate_puzzle(self):
		"""generates a new puzzle and solves it"""
		self.generate_solution(self.grid)
		#self.print_grid('full solution')
		self.remove_numbers_from_grid()
		#self.print_grid('with removed numbers')
		return

	def print_grid(self, grid_name=None):
		if grid_name:
			print(grid_name)
		for row in self.grid:
			print(row)
		return

	def test_sudoku(self,grid):
		"""tests each square to make sure it is a valid puzzle"""
		for row in range(4):
			for col in range(4):
				num = grid[row][col]
				#remove number from grid to test if it's valid
				grid[row][col] = 0
				if not self.valid_location(grid,row,col,num):
					return False
				else:
					#put number back in grid
					grid[row][col] = num
		return True

	def num_used_in_row(self,grid,row,number):
		"""returns True if the number has been used in that row"""
		if number in grid[row]:
			return True
		return False

	def num_used_in_column(self,grid,col,number):
		"""returns True if the number has been used in that column"""
		for i in range(4):
			if grid[i][col] == number:
				return True
		return False

	def num_used_in_subgrid(self,grid,row,col,number):
		"""returns True if the number has been used in that subgrid/box"""
		sub_row = (row // 2) * 2
		sub_col = (col // 2)  * 2
		for i in range(sub_row, (sub_row + 2)): 
			for j in range(sub_col, (sub_col + 2)): 
				if grid[i][j] == number: 
					return True
		return False

	def valid_location(self,grid,row,col,number):
		"""return False if the number has been used in the row, column or subgrid"""
		if self.num_used_in_row(grid, row,number):
			return False
		elif self.num_used_in_column(grid,col,number):
			return False
		elif self.num_used_in_subgrid(grid,row,col,number):
			return False
		return True

	def find_empty_square(self,grid):
		"""return the next empty square coordinates in the grid"""
		for i in range(4):
			for j in range(4):
				if grid[i][j] == 0:
					return (i,j)
		return

	def solve_puzzle(self, grid):
		"""solve the sudoku puzzle with backtracking"""
		for i in range(0,16):
			row=i//4
			col=i%4
			#find next empty cell
			if grid[row][col]==0:
				for number in range(1,5):
					#check that the number hasn't been used in the row/col/subgrid
					if self.valid_location(grid,row,col,number):
						grid[row][col]=number
						if not self.find_empty_square(grid):
							self.counter+=1
							break
						else:
							if self.solve_puzzle(grid):
								return True
				break
		grid[row][col]=0  
		return False

	def generate_solution(self, grid):
		"""generates a full solution with backtracking"""
		number_list = [1,2,3,4]
		for i in range(0,16):
			row=i//4
			col=i%4
			#find next empty cell
			if grid[row][col]==0:
				shuffle(number_list)      
				for number in number_list:
					if self.valid_location(grid,row,col,number):
						self.path.append((number,row,col))
						grid[row][col]=number
						if not self.find_empty_square(grid):
							return True
						else:
							if self.generate_solution(grid):
								#if the grid is full
								return True
				break
		grid[row][col]=0  
		return False

	def get_non_empty_squares(self,grid):
		"""returns a shuffled list of non-empty squares in the puzzle"""
		non_empty_squares = []
		for i in range(len(grid)):
			for j in range(len(grid)):
				if grid[i][j] != 0:
					non_empty_squares.append((i,j))
		shuffle(non_empty_squares)
		return non_empty_squares

	def remove_numbers_from_grid(self):
		"""remove numbers from the grid to create the puzzle"""
		#get all non-empty squares from the grid
        # 
		non_empty_squares = self.get_non_empty_squares(self.grid)
		non_empty_squares_count = len(non_empty_squares)
		rounds = 3
		while rounds > 0 and non_empty_squares_count >= 4:
			#there should be at least 17 clues
			row,col = non_empty_squares.pop()
			non_empty_squares_count -= 1
			#might need to put the square value back if there is more than one solution
			removed_square = self.grid[row][col]
			self.grid[row][col]=0
			#make a copy of the grid to solve
			grid_copy = copy.deepcopy(self.grid)
			#initialize solutions counter to zero
			self.counter=0      
			self.solve_puzzle(grid_copy)   
			#if there is more than one solution, put the last removed cell back into the grid
			if self.counter!=1:
				self.grid[row][col]=removed_square
				non_empty_squares_count += 1
				rounds -=1
		return


def make_4X4_puzzles(i:int = 10):
    """
    Function that returns a dictionary
    with integeger keys 0 through i.
    The values are a tuple of two lists of lists.
    the first is the 4X4 puzzle unsolved grid,
    the second is the unique solution.
    """
    puzzle4X4_dict = dict()
    for j in range(i):
        new_puzzle = SudokuGenerator4X4()
        dfpuzzle4X4 = pd.DataFrame.from_records(new_puzzle.grid)
        new_puzzle_sol = SudokuGenerator4X4(grid=dfpuzzle4X4.to_numpy().tolist())
        puzzle4X4_dict[j] = (new_puzzle.grid, new_puzzle_sol.grid)
    return puzzle4X4_dict


def make_9X9_puzzles(i:int = 10):
    """
    Function that returns a dictionary
    with integeger keys 0 through i.
    The values are a tuple of two lists of lists.
    the first is the 4X4 puzzle unsolved grid,
    the second is the unique solution.
    """
    puzzle9X9_dict = dict()
    for j in range(i):
        new_puzzle = SudokuGenerator9X9()
        dfpuzzle9X9 = pd.DataFrame.from_records(new_puzzle.grid)
        new_puzzle_sol = SudokuGenerator9X9(grid=dfpuzzle9X9.to_numpy().tolist())
        puzzle9X9_dict[j] = (new_puzzle.grid, new_puzzle_sol.grid)
    return puzzle9X9_dict




os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# we need a streamlit input form that picks the 
# puzzle type :4X4 or 9X9
# the number of few shot examples k, an integer
# the openai model, one of gpt-4 , gpt-4-1106-preview'
# the temperature, a float between 0 and 2

# maybe add another option like "Be encouraging?" and then change the prompt appropriately.


def get_examples(new_puzzle:str, rqa):
    """
    takes in a new_puzzle with format like
    [[0, 0, 4, 0], [0, 0, 0, 0], [0, 3, 0, 0], [1, 0, 0, 2]]
    and returns the string of examples that will
    go into the template_string
    """
    results = rqa({"query":new_puzzle})
    examples = results['source_documents']
    eg_string = ''
    for eg in examples:
        eg_string += eg.page_content + '\n'
    return eg_string



template_string_4X4 =  """
role: You An expert Sudoku Puzzle solver that can solve 9X9 puzzles as well as 4X4 puzzles.
content: Here are some example puzzles and their solutions.

{example_solutions}

The rules of 4x4 Sudoku puzzles are the same as with traditional Sudoku grids.
Only the number of cells and digits to be placed are different.
1. The numbers 1, 2, 3 and 4 must occur only once in each column
2. The numbers 1, 2, 3 and 4 must occur only once in each row.
3. The clues allocated at the beginning of the puzzle cannot be changed or moved.

Each row and column must contain each of the numbers 1-4. 
Each of the four corner areas of 4 cells must also contain each of the numbers 1-4.



task: Using the rules of Sudoku, solve the initial 4X4 grid below. 0 indicates a missing
digit needing to be filled in. Think Step by Step. I will pay you $5,000 dollars for a correct solution.
Break it down carefully. Think logically and carefully. Each step in your solution
should be correct and obvious logically. No erasers needed for your expertise!
{puzzle}

The rules of 4x4 Sudoku puzzles are the same as with traditional Sudoku grids.
Only the number of cells and digits to be placed are different.
1. The numbers 1, 2, 3 and 4 must occur only once in each column
2. The numbers 1, 2, 3 and 4 must occur only once in each row.
3. The clues allocated at the beginning of the puzzle cannot be changed or moved.

Did you answer the previous question correctly? Your previous answer was {previous}. The actual solution is {answer}.
This is the puzzle to be solved:
{puzzle}

Work this out in a step-by-step way. Take your time.

{format_instructions}
"""



template_string_9X9 =  """
role: You An expert Sudoku Puzzle solver that can solve 9X9 puzzles as well as 4X4 puzzles.
content: Here are some example puzzles and their solutions.

{example_solutions}

The rules of 9x9 Sudoku puzzles are the same as with traditional Sudoku grids.
Only the number of cells and digits to be placed are different.
1. EAch row hsould have numbers 1-9, no repeats.
2. Each column should have numbers 1-9, no repeats.
3. Each 3X3 quadrant should have numbers 1-9, no repeats.
3. The clues allocated at the beginning of the puzzle cannot be changed or moved.




task: Using the rules of Sudoku, solve the initial 9X9 grid below. 0 indicates a missing
digit needing to be filled in. Think Step by Step. I will pay you $5,000 dollars for a correct solution.
Break it down carefully. Think logically and carefully. Each step in your solution
should be correct and obvious logically. No erasers needed for your expertise!
{puzzle}



Did you answer the previous question correctly? Your previous answer was {previous}. The actual solution is {answer}.
This is the puzzle to be solved:
{puzzle}

Work this out in a step-by-step way. Take your time.

{format_instructions}
"""




with st.form("my_form"):
    st.write("Input parameters:")
    puzzle_type = st.selectbox('Pick the puzzle type', ['4X4','9X9'])
    model = st.selectbox('Pick an OpenAI model', ['gpt-4', 'gpt-4-1106-preview','gpt-4',
                                                 'gpt-3.5-turbo-1106'])
    k = st.slider('Pick the number of examples to show the model in the instructions', 1, 10)
    j = st.slider('Pick the number of puzzles to ask the model to try', 2, 20)
    temperature = st.number_input('Select the model temperature', min_value=0., max_value=2., value = 0.0, step=0.01)
    submit = st.form_submit_button("Run experiment")


# this is outside the form
#submit = my_form.form_submit_button('Run experiment')
if submit:
    st.write("Preparing the experiment:")

    if puzzle_type == '4X4':
        with open("puzzle4X4_dict.pkl", "rb") as f:
            puzzle_dict = pickle.load(f)
        docsearch = FAISS.load_local("faiss_index_4X4", embeddings) # might be able to cache this
        test_dict = make_4X4_puzzles(i=j+2)
        template_string = template_string_4X4[:]
    if puzzle_type == '9X9':
        with open("puzzle9X9_dict.pkl", "rb") as f:
            puzzle_dict = pickle.load(f)
        docsearch = FAISS.load_local("faiss_index_9X9", embeddings) # might be able to cache this
        test_dict = make_9X9_puzzles(i=j+2)
        template_string = template_string_9X9[:]
    embeddings = OpenAIEmbeddings()
    #docsearch4X4 = FAISS.load_local("faiss_index_4X4", embeddings) # might be able to cache this
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":k})

    # create the chain to answer question

    rqa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name= model),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)
    #test_dict = make_4X4_puzzles(i=j+2)
    
    chat_llm = ChatOpenAI(model_name = model, temperature=temperature)
#chat_llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature=0)

    puzzle_name_schema = ResponseSchema(name="input_puzzle",
                             description="This is the sudoku puzzle to be solved")

    solution_schema = ResponseSchema(name="solution",
                                      description="This is the puzzle solution")

    reasoning_schema = ResponseSchema(name="reasoning",
                                    description="This is the reasons for the solution")

    confidence_schema = ResponseSchema(name="confidence",
                                   description="This is the confidence in the solution, a number between 0 and 1.")

    response_schemas = [puzzle_name_schema,
                    solution_schema,
                    reasoning_schema,
                   confidence_schema]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_template(template=template_string)
    def get_messages(i:int, format_instructions:str=format_instructions, previous:str='correct'):
        """
        i is the index of the test_dict solved puzzle dictionary.
        Returns the prompt.format_messages result
        where the puzzle to be solbed corresponds
        to the ith key of the puzzle dictionary.
        """
        messages = prompt.format_messages(
            example_solutions = get_examples(str(test_dict[i][0]), rqa), # i = 0
            puzzle = test_dict[i][0],
            previous=previous,
            answer=test_dict[i-1],
          #  input_puzzle=test_dict[i][0],
            format_instructions=format_instructions    
        )
        return messages
    
    llm = ChatOpenAI(model_name = model, temperature=temperature)
    #llm = ChatOpenAI(model_name = "gpt-4", temperature=0)
    # we set a low k=2, to only keep the last 2 interactions in memory
    #llm = ChatOpenAI(model_name = "davinci-002", temperature=0.2)
    if puzzle_type == '4X4':
        window_memory = ConversationBufferWindowMemory(k=2)
    if puzzle_type == '9X9':
        window_memory = ConversationBufferWindowMemory(k=1)

    conversation = ConversationChain(
        llm=llm, 
       # llm=llm_llama2,
        verbose=True, 
        memory=window_memory
    )
    results = []
    for i in range(1, j+1):
        if len(results) > 0:
            last = results[-1]
            if last == True:
                previous = 'correct'
            else:
                previous = 'incorrect'
        else:
            previous = 'correct'
   # example_solutions = get_examples(str(test_dict[i][0])) # i = 0
        messages =  get_messages(i=i, previous=previous)
        response = conversation.predict(input=messages[0].content)
        st.write(messages[0].content)
        st.write(response)
        try:
            response_as_dict = output_parser.parse(response)
            res = test_dict[i][1] == ast.literal_eval(response_as_dict["solution"])
            print(test_dict[i][1], ast.literal_eval(response_as_dict["solution"]))
            print(res, i, "can use output_parser", response_as_dict["confidence"])
            dfactual = pd.DataFrame.from_records(test_dict[i][1])
            dflmm = pd.DataFrame.from_records(ast.literal_eval(response_as_dict["solution"]))
            #st.write(test_dict[i][1])
            dfpuzzle = pd.DataFrame.from_records(test_dict[i][0])
            st.markdown('### Puzzle')            
            # style
            th_props = [
                  ('font-size', '14px'),
                  ('text-align', 'center'),
                  ('font-weight', 'bold'),
                  ('color', '#FF0000'),
              #    ('background-color', '#f7ffff')
            ]
            td_props = [
                  ('font-size', '12px')
              ]
            styles = [
              dict(selector="th", props=th_props),
              dict(selector="td", props=td_props)
              ]
           # dftablepuzzle = dfpuzzle.style.set_table_styles(styles)
            dfpuzzlecolor = dfpuzzle.style.set_table_styles(
                       [{
                           'selector': 'th',
                       'props': [('color', 'red')]
                       }]
            )
           # st.table(dftablepuzzle)
            st.dataframe(dfpuzzle, hide_index=True)
          #  st.markdown(dfpuzzle.set_properties(**{'color': '#F0000',
           #                 'font-weight': 'hold'}.hide(axis=0).hide(axis=1).to_html(),
           #                                     unsafe_allow_html=True))
           # rows[0].markdown('|'.join(dfpuzzle.to_markdown(index=False).replace('0', '').split('|')[5:])[1:])
            rows = st.columns(2)
            rows[0].markdown('### Actual Solution')
            rows[0].dataframe(dfactual, hide_index=True)
            rows[1].markdown('### LLM Solution')
            rows[1].dataframe(dflmm, hide_index=True)
            st.markdown('### LLM reasoning / chain-of-thought')
            st.write(response_as_dict["reasoning"])
            #st.write(dfactual)
            #st.write("LLM solution:")
            #st.write(ast.literal_eval(response_as_dict["solution"]))
            results.append(res)
        except:
            res = str(test_dict[i][1])  in response
            #print(puzzle4X4_dict[i][1], ast.literal_eval(response_as_dict["solution"]))
            #print(response)
            #st.write(response)
            print(res, i, "can not use output_parser")
            dfactual = pd.DataFrame.from_records(test_dict[i][1])
            #st.write("actual solution:")
            #st.write(test_dict[i][1])
            solution_matches = re.findall(r'"solution": "(.*?)"', response)
            reasoning_matches = re.findall(r'"reasoning": "(.*?)"', response)
            last_solution = solution_matches[-1] if solution_matches else None
            last_reasoning = reasoning_matches[-1] if reasoning_matches else None
            dflmm = pd.DataFrame.from_records(ast.literal_eval(last_solution))
            dfpuzzle = pd.DataFrame.from_records(test_dict[i][0])
            st.markdown('### Puzzle')   
            st.dataframe(dfpuzzle, hide_index=True)
            rows = st.columns(2)
            rows[0].markdown('### Actual Solution')
            rows[0].dataframe(dfactual, hide_index=True)
            rows[1].markdown('### LLM Solution')
            rows[1].dataframe(dflmm, hide_index=True)
            st.markdown('### LLM reasoning / chain-of-thought')
            st.write(last_reasoning)
            
           # st.write("LLM solution:")
           # st.write(ast.literal_eval(response_as_dict["solution"]))
            results.append(res)
        print(pd.Series(results).value_counts())
        st.write(pd.Series(results).value_counts())

    print(pd.Series(results).value_counts())
    st.write("Final results:")
    st.write(pd.Series(results).value_counts())




