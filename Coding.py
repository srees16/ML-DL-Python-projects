'''Code Practice'''

https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search
https://analyticsindiamag.com/why-is-random-search-better-than-grid-search-for-machine-learning


Vanishing gradient problem can be fixed by:
    1. using right activation functions
    2. using Residual networks or RNN like LSTM, GRU
    3. using batch normalization

rank, over, partition by in postgresql
https://www.postgresqltutorial.com/postgresql-rank-function
https://www.postgresqltutorial.com/postgresql-window-function

partition by table
rank over
GCP USES BIGQUERY

confidence interval

# https://www.codesdope.com/practice/python-make-a-list

# Take 10 integer inputs from user and store them in a list and print them on screen.
def inputs():
    nos = []
    no_of_inputs = 10
    while no_of_inputs > 0:
        print('Enter a number ')
        val = int(input())
        nos.append(val)
        no_of_inputs -= 1
    return nos

inputs()

# improve the above wit try except
def inputs():
    try:
        nos = []
        no_of_inputs = 10
        while no_of_inputs > 0:
            print('Enter a number ')
            val = int(input())
            nos.append(val)
            no_of_inputs -= 1
        return nos
    except:
        print('Your input is not an integer, please try again!')
        print('Enter a number ')
        val = int(input())
        continue

inputs()
'''Take 10 integer inputs from user and store them in a list. Again ask user to give a number. Now, tell user
whether that number is present in list or not'''
def number_present():
    number_of_inputs = 10
    nos = []
    while number_of_inputs > 0:
        print('Enter 10 numbers ')
        val = int(input())
        nos.append(val)
        number_of_inputs -= 1
    print('Enter a number to search in your previous input: ')
    n = int(input())
    if n in nos:
        print('Hooray!')
    else:
        print('Oops!!')

number_present()

my_lis = [(4, 3), (1, 2), (6, 1)]

sorted(my_lis, key = lambda x: x[1]) # Or
my_lis.sort(key = lambda x: x[1])
my_lis

for x in my_lis:
    for x[0] in x:
        if y < x:
            print (y)
sorting(my_lis)

'''Prime nos'''
Find the sum of all the primes below two million.
primes upto 10 = 2 + 3 + 5 + 7
sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.

reasoning: 2/1 or 2/2
1. x % 2 to x-1 == 0
2. add them all to a list
3. return sum of the list
4. answer 74 for input 20

def multiples(n):
    multiples_list = []
    upto = list(range(2, n))
    for x in upto:
        if (n % x) == 0:
            multiples_list.append(x)
    return multiples_list

timeit multiples(20)

'''prime nos upto N'''
def primes(n):
    primes_list = []
    for x in range(2, n):
        for i in range(2, x):
            if x % i == 0:
                break
        else:
            primes_list.append(x)
    return primes_list

primes(20)
timeit primes(20)

'''Find Nth prime no. What is 100th prime no? 6th prime is 13 for the first six prime numbers: 2, 3, 5, 7, 11, 13'''
def primes(n):
    primes_list = []
    for x in range(2, n+1):
        for i in range(2, x):
            if x % i == 0:
                break
        else:
            primes_list.append(x)
    return primes_list[-1]

'''Eval function'''
def create_function():
    expr = input('Enter a mathematical expression for "x": ')
    inp = int(input('Enter x: '))
    func = eval(expr)
    res = func(inp)
    return res

create_function()

# pass any no of args n perform aggregation fuctions for those. Hint: use eval function & *args
maths = ['sum', 'len', 'np.min', 'np.max', 'np.mean']
division = ['modulus', 'divide']

myfunc = eval('np.mean')
myfunc([2,1,5])

def math_ops(*args):
    results = []
    for x in maths:
        func = eval(x)
        results.append(func(args))
    return results

math_ops(7,10,4,6)

games = ['heads', 'heads', 'tails', 'heads', 'tails']
games2 = ['heads', 'heads', 'heads', 'heads', 'heads']

# find which list has unique elements
len(games) == len(set(games)) # simple boolean check
len(games2) == len(set(games2))

# find most common item in a list
games.count('heads')
items = set(games)
max(items, key = games.count)

'''
Take 20 integer inputs from user and print the following:
number of positive numbers
number of negative numbers
number of odd numbers
number of even numbers
number of 0s.
'''
def number_present():
    number_of_inputs = 10
    input_list = []
    positives = []
    negatives = []
    odds = []
    evens = []
    zeros = []
    while number_of_inputs > 0:
        print('Enter 10 numbers ')
        val = int(input())
        input_list.append(val)
        number_of_inputs -= 1
    for i in input_list:
        if i >= 0:
            positives.append(i)
            if i == 0:
                zeros.append(i)
            if i % 2 != 0:
                odds.append(i)
            elif i % 2 == 0:
                evens.append(i)
        else:
            negatives.append(i)
    return positives, zeros, negatives, odds, evens
number_present()

# Take 10 integer inputs from user and store them in a list. Now, copy all the elements in another list but in reverse order.
def rev_nos():
    nos = []
    rev_no = []
    no_of_inputs = 10
    while no_of_inputs > 0:
        print('Enter 10 numbers: ')
        vals = int(input())
        nos.append(vals)
        no_of_inputs -= 1
        rev_no = nos[::-1].copy()
    return rev_no
rev_nos()

# Write a program to find the sum of all elements of a list.
def summation1():
    from functools import reduce
    nos = [1, 2, 3, 4, 5]
    print(reduce(lambda x, y: x + y, nos))
summation1()

def summation2(nos):
    return sum(nos)
summation2([1, 2, 3, 4, 5])

def summation3(*args):
    tot = 0
    for element in range(0, len(args)):
        tot = tot + args[element]
    return tot
summation3(1, 2, 3, 4, 5, 10)

# Write a program to find the product of all elements of a list.
def multiples(*args):
    from functools import reduce
    return reduce(lambda x, y: x * y, args)
multiples(1, 2, 3, 4, 5)

def multiples1(nos):
    prod = 1
    for element in range(0, len(nos)):
        prod = nos[element] * prod
    return prod
multiples1([1, 2, 3, 4, 5])

# Initialize and print each element in new line of a list inside list

# Find largest and smallest elements of a list
w = [10, 9, 6, 5, 3, 1, 12, 8, 13]  #for range(list), ele returns index, w[ele] gives elements
def large_small(ls):
    small = ls[0]
    big = ls[0]
    for ele in range(len(ls)):
        if ls[ele] < small:
            small = ls[ele]
            ele += 1
        if ls[ele] > big:
            big = ls[ele]
            ele += 1
    return small, big
large_small(w)

# Using reduce
from functools import reduce
max_find = lambda x, y: x if x > y else y
min_find = lambda x, y: x if x < y else y
reduce(max_find, w)
reduce(min_find, w)

# Using aggregators
max(w)
min(w)

# Write a program to print sum, average of all numbers, smallest and largest element of a list.
def list_details(s):
    avg = 0
    tot = 0
    small = s[0]
    big = s[0]
    for element in range(len(s)):
        tot = tot + s[element]
        element += 1
        avg = tot / len(s)
    for element in range(len(s)):
        if s[element] < small:
            small = s[element]
            element += 1
        if s[element] > big:
            big = s[element]
            element += 1
    return tot, round(avg, 2), small, big
list_details(w)

# Write a program to check if elements of a list are same or not if read from front or back
x = [2, 3, 15, 15, 3, 2]
def palindrome(liss):
    frontal = liss[::]
    back = liss[::-1]
    return frontal == back
palindrome(x)

def palindromes(lis):
    for front_order_element in lis:
        for back_order_element in lis[::-1]:
            if front_order_element == back_order_element:
                return True
palindromes(x)

def palindrome2(lis):
    i = 0
    mid = len(lis) / 2
    same = True
    while i < mid:
        if lis[i] != lis[len(lis) - i - 1]:
            print('No')
            same = False
            break
        i += 1
    if same == True:
        print('Yes')
palindrome2(x)

# Take a list of 10 elements. Split it into middle and store the elements in two dfferent lists
tes = [58, 24, 13, 15, 63, 9, 8, 81, 1, 78]
def list_split(x):
    first_half = []
    next_half = []
    half_1 = x[0 : int(len(x) / 2)]
    half_2 = x[int(len(x) / 2) :]
    for ele in half_1:
        first_half.append(ele)
    for eles in half_2:
        next_half.append(eles)
    return first_half, next_half
list_split(tes)

# Ask user to give integer inputs to make a list. Store only even values given and print the list
def only_evens():
    threshold = 10
    inputs = []
    evens = []
    while threshold > 0:
        print('Enter 10 numbers of your choice: ')
        val = int(input())
        threshold -= 1
        inputs.append(val)
    for ele in range(len(inputs)):
        if inputs[ele] % 2 == 0:
            evens.append(inputs[ele])
            ele += 1
    return evens
only_evens()

mat1 = np.random.randint(10, 15, 9).reshape(3, 3)
mat2 = np.random.randint(16, 56, 9).reshape(3, 3)
mat1 + mat2
mat1 * mat2
np.random.randn(1, 5, 4).shape

# Make a list by taking 10 input from user. Now delete all repeated elements of the list To do
s = [1,2,3,2,1,3,12,12,32,2] # input
[1,2,3,12,32] # output

def duplicate_deletion(lis):
    new_list = []
    for x in range(len(lis)):
        for y in range(len(lis)):
            if lis[x] == lis[y]:
                lis.pop(x)
        return lis

s = [1,2,3,2,1,3,12,12,32,2] # input
duplicate_deletion(s)

# Given two integers return their product and if the product is greater than 1000, then return their sum
def prod_sum(*args):
    prod = np.multiply(*args)
    sums = sum(args)
    if prod > 1000:
        return sums
    else:
        return prod

prod_sum(50, 3, 870)

# Given a range of first 10 nos, iterate from start no to end no and print sum of current no and previous no
def no_iter(n):
    sums = 0
    prev = 0
    for i in range(n):
        sums += i
        print('current no {} previous no {} sum: {}'.format(i, prev, sums))
        prev += i

no_iter(5)

# Given a string, display only those characters which are present at an even index number
string = 'pynative'
def even_chars(s):
    even_chars_lis = []
    for i in range(len(s)):
        if i % 2 == 0:
            even_chars_lis.append(s[i])
    return even_chars_lis

even_chars(string)

# Given a string and an integer number n, remove characters from a string starting from 0 up to n, return a new string
string = 'current_pynative_sum'
def remove_chars(strng, num):
    new_string = strng[num+1:]
    return new_string

remove_chars(string, 6)

# Given a list of numbers, return True if first and last number of a list is same
l1 = [10, 20, 30, 40, 10]
l2 = [10, 20, 30, 40, 50]

def last_1st(lis):
    first = lis[0]
    last = lis[-1]
    if first == last:
        return True
    return False

last_1st(l1)
last_1st(l2)

# Given a list of numbers, iterate it and print only those numbers which are divisible of 5
ls = [10, 20, 33, 46, 55]

def factors_5(lis):
    factors = []
    for x in lis:
        if x % 5 == 0:
            factors.append(x)
    return factors

factors_5(ls)

# Return the total count of sub-string “Emma” appears in the given string
string = 'Emma is good developer. Emma is a writer'

def word_count(strs):
    count = 0
    for x in strs.split():
        if x == 'Emma':
            count += 1
    return count

word_count(string)

# Print the following pattern to do
1
2 2
3 3 3
4 4 4 4
5 5 5 5 5

# Print the following pattern using for loop to do
5 4 3 2 1
4 3 2 1
3 2 1
2 1
1

# Print downward Half-Pyramid Pattern with Star (asterisk)
* * * * *
* * * *
* * *
* *
*

# Accept number from user and calculate the sum of all number between 1 and given number
def sum_until():
    tot = 0
    print('Enter number you want sum upto: ')
    n = int(input())
    for i in range(1, n+1):
        tot += i
    return tot

sum_until()

# multiplication table of given number
def multipli_table(n):
    for x in range(1, 11):
        print(n * x)
multipli_table(2)

# Given a list iterate and display numbers divisible by 5, if number greater than 150 stop the loop iteration
list1 = [12, 15, 32, 42, 55, 75, 122, 132, 150, 180, 200]

def multiple_5(lis):
    for x in lis:
        if x % 5 == 0:
            if x <= 150:
                print(x)
            else:
                break

multiple_5(list1)

# Given a number count the total number of digits in a number
def digits(n):
    m = str(n)
    return len(m)

digits(x)

# Reverse the following list using for loop
list1 = [10, 20, 30, 40, 50]
def reverse_for(lis):
    rev_list = []
    for x in lis[::-1]:
        rev_list.append(x)
    return rev_list

reverse_for(list1)

# Display -10 to -1 using for loop
for i in range(-10, 0):
    print(i)

'''Given a two list of nos create new list that contains only odd numbers from 1st list and even numbers from 2nd list
fl = [10, 20, 23, 11, 17]
sl = [13, 43, 24, 36, 12]
rl = [23, 11, 17, 24, 36, 12]'''

def odd_even_list(l1, l2):
    new_l = []
    new_l.extend([x for x in l1 if x % 2 != 0])
    new_l.extend([x for x in l2 if x % 2 == 0])
    return new_l

odd_even_list(fl, sl)

# extract each digit from an integer, in the reverse order. ex: int 7536, output be '6 3 5 7'
def reverse_ints(n):
    x = str(n)
    return list(x[::-1])

reverse_ints(7536)

# Print multiplication table form 1 to 10
def table():
    for i in range(1, 11):
        for j in range(1, 11):
            print(i * j, end = ' ')
        print('\t')
table()

# check if file is empty using context manager
def file_empty():
    with open('url.txt', 'r') as f:
        lines = f.readline()
        if len(lines) == 0:
            return True
        else:
            return False
file_empty()

# Accept two numbers from the user and calculate multiplication
def multiply():
    print('Enter 2 numbers: ')
    i = 2
    inputs = []
    while i > 0:
        val = int(input())
        inputs.append(val)
        i -= 1
    prod = 1
    for i in range(len(inputs)):
        prod *= inputs[i]
    return prod
multiply()

# for n no of args
def muls(*args):
    prod = 1
    for i in range(len(args)):
        prod = prod * args[i]
    return prod
muls(1,2,3,4)

# display “My Name Is James” as “My**Name**Is**James” using output formatting of a print() function
def mis_display():
    print('enter any sentence: ')
    inp = str(input())
    for i in inp.split(' '):
        print(i, end = '**')

mis_display()
# or
print('My', 'Name', 'Is', 'James', sep = '~')

'''Calculate income tax for the given income by adhering to the below rules
Taxable Income	Rate (%)
First $10,000	0
Next $10,000	10
The remaining	20
Ex: taxable income is $45000 the income tax payable is
$10000*0% + $10000*10%  + $25000*20% = $6000'''

1. income is 67000
2. subtract 10000 that has no IT
3. subtract next 10000 that has 10% IT
4. calculate IT on remaining income

def income_tax(income):
    no_slab = 10000
    tenpc_slab = 10000
    it_slab = income - no_slab - tenpc_slab
    if income <= no_slab:
        payable_tax_0 = 0
    elif no_slab <= income <= tenpc_slab:
        payable_tax_10 = 0.1
    elif income > no_slab + tenpc_slab:
        payable_tax_20 = it_slab
        #print(no_slab, tenpc_slab, it_slab)
        income_tax = tenpc_slab * .1 + payable_tax_20 * .2
    return income_tax

income_tax(67000)

def income_tax_indian(income):
    no_slab = 250000
    fivepc_slab = 250000
    tenpc_slab = 250000
    fifteenpc_slab = 250000
    twentypc_slab = 250000
    twentyfivepc_slab = 250000
    thirtypc_slab = income - twentyfivepc_slab - twentypc_slab - fifteenpc_slab - tenpc_slab - fivepc_slab - no_slab
    if income <= no_slab:
        payable_tax_0 = 0
    elif no_slab <= income <= fivepc_slab:
        payable_tax_5 = 0.05
    elif fivepc_slab <= income <= tenpc_slab:
        payable_tax_10 = 0.1
    elif tenpc_slab <= income <= fifteenpc_slab:
        payable_tax_15 = 0.15
    elif fifteenpc_slab <= income <= twentypc_slab:
        payable_tax_20 = 0.2
    elif twentypc_slab <= income <= twentyfivepc_slab:
        payable_tax_25 = 0.25
    elif twentyfivepc_slab <= income <= thirtypc_slab:
        payable_tax_30 = 0.3
        print(no_slab, fivepc_slab, tenpc_slab, fifteenpc_slab, twentypc_slab, twentyfivepc_slab, thirtypc_slab)
        income_tax = fivepc_slab * payable_tax_5 + tenpc_slab * payable_tax_10 + fifteenpc_slab * payable_tax_15 + twentypc_slab * payable_tax_20 + twentyfivepc_slab * payable_tax_25 + thirtypc_slab * payable_tax_30
    return income_tax


s1, s2, s3, s4 = input('enter string for splitting ').split()
print(s1, s2, s3, s4)

# write all file content into new file by skiping line 5 from following file To Do


# find specific word from a string
string = 'Coding Practice Pandas Numpy EDA Visualization'
def find_word(s):
    for i in s.split():
        if i.casefold() == 'eda':
            return i

find_word(string)

def word_catcher(line):
    import re
    pattern = re.compile('\s[A-Z]+\s')
    match = pattern.finditer(line)
    x = [x[0] for x in match]
    return x

word_catcher(string)

# remove a specific word from a string
def delete_word(sentence, word):
    words = []
    for i in sentence.split():
        words.append(i)
    if word in words:
        words.remove(word)
        return ' '.join(words)

delete_word(string, 'Coding')

# remove multiple specific words from a sentence and return it
def delete_words(sentence, *args):
    words = []
    for i in sentence.split():
        words.append(i)
    for a in args:
        if a in words:
            words.remove(a)
    return ' '.join(words)

delete_words(string, 'Coding', 'EDA', 'Numpy' )

# Accept any three string from one input() call
def inputs_3():
    print('enter 3 strings')
    strs = 3
    while strs > 0:
        (s1, s2, s3) = input().split(' ')
        strs -= 1
        return s1, s2, s3
inputs_3()

# Accept n no of strings from one input() call & print them
def specified_inputs():
    lis = []
    print('How many strings you wanna enter? ')
    val = int(input())
    while val > 0:
        print('Enter string: ')
        inp = input()
        lis.append(inp)
        val -= 1
    return lis

specified_inputs()

# Accept several inputs from one input() call as a tuple & print them
def several_inputs():
    print('Enter any no of strings you want!')
    (*args, ) = input()
    joined_args = ''.join(args)
    return joined_args

several_inputs()

# improve above method for taking n no of strings into single tuple variable
def n_inputs(*args):
    li = list(args)
    return li
n_inputs(2,3,6,'asdf')

def n_inputs():
    print('enter any no of strings: ')
    (*args,) = input()
    return args
    while len(arg) > 0:
        print(*args)
        arg -= 1

n_inputs()

# return list with words joined by _
houses = ["Eric's house", "Kenny's house", "Kyle's house", "Stan's house"]
def join_houses(lis):
    new_houses = []
    for l in lis:
        x = l.split(' ')
        new_houses.append('_'.join(x))
    return new_houses

join_houses(houses)

# requests library https://realpython.com/python-requests/
import requests
x = requests.get('https://w3schools.com/python')
x.text
x.status_code
x.json()
x.headers
x.headers['Content-Encoding']
x.headers['content-type']
'''
delete(url, args): sends a DELETE request to the specified url
get(url, params, args): sends a GET request to the specified url
head(url, args): sends a HEAD request to the specified url
patch(url, data, args): sends a PATCH request to the specified url
post(url, data, json, args): sends a POST request to the specified url
put(url, data, args): sends a PUT request to the specified url
request(method, url, args):	sends a request of the specified method to the specified url
'''

# Create an outer function to accept two parameters. Create inner function to sum them. return with addition of 5 & assign a
# different name to function and call it through the new name
def outside(*args):
    def inside():
        return sum(args)
    return 5 + inside()

outside_func = outside
outside_func(4,5,6)

# Given a string of odd length greater than 7, return string made of the middle three chars of a given String
getMiddleThreeChars("JhonDipPeta") → 'Dip'
getMiddleThreeChars("Jasonay") → 'son'
def get_middle_three_chars(word):
    mid_index = int(len(word) / 2)
    mid_three = word[mid_letter - 1: mid_letter + 2]
    return mid_three
word = 'Commented'
get_middle_three_chars(word)

# Given 2 strings, s1 and s2, create a new string by appending s2 in the middle of s1
def mid_append(word1, word2):
    mid_index = int(len(word1) / 2)
    first_half = word1[0 : mid_index]
    sec_half = word1[mid_index:]
    new = first_half + word2 + sec_half
    return new
mid_append('Chrisdem', 'IamNewString')

# Given 2 strings, s1, and s2 return a new string made of the first, middle and last char each input string
mixString("America", "Japan") = "AJrpan"

def mixString(w1, w2):
    mid_w1 = int(len(w1) / 2)
    mid_w2 = int(len(w2) / 2)
    first_w1 = w1[0]
    first_w2 = w2[0]
    last_w1 = w1[-1]
    last_w2 = w2[-1]
    mixy = first_w1 + first_w2 + w1[mid_w1] + w2[mid_w2] + last_w1 + last_w2
    return mixy

mixString("America", "Japan")

def add():
    input_size = 5
    nos = []
    tot = 0
    while input_size > 0:
        print('enter 2 numbers: ')
        vals = int(input())
        nos.append(vals)
        input_size -= 1
    for i in range(len(nos)):
        tot += nos[i]
        i += 1
    return tot

add()

lis = [4,3,5,8,1,6]
def adds(lis):
    tot = 0
    for i in lis:
        tot = tot + i
        print(tot)

adds(lis)

# get list of strings having specific character
df = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\adult.csv')
cols = pd.Series(df.columns)
under_score_titles = cols[cols.str.contains('-')]
under_score_titles

national = ['India', 'US', 'Nepal', 'Bhutan', 'Ja_pan', 'UK', 'Fra_nce', 'Germany', 'Croatia']
# method 1
r = re.compile('.*_.*')
list(filter(r.match, national))
# method 2
def search_this(lis):
    nats= []
    for nation in range(len(lis)):
        for letter in lis[nation]:
            if letter == '_':
                nats.append(lis[nation])
    return nats

search_this(national)

listr = ['hello_dear', 'hello_yaar', 'hello_my_dear']
def find_():
    for i in listr:
        return re.search('hello', lamda x: x[i] for i in x) # fix this
find_()


''''''''''''
df = pd.read_csv(r'C:\Users\Srees\Desktop\Dataset\Train.csv')
df.sample(4)
df.shape
df.info()
from datetime import date
dates = pd.to_datetime(df['DATE'])

'''You are driving a little too fast, and a police officer stops you. Write a function to return one of 3 possible results: "No ticket", "Small ticket", or "Big Ticket". If your speed is 60 or less, the result is "No Ticket". If speed is between 61 and 80 inclusive, the result is "Small Ticket". If speed is 81 or more, the result is "Big Ticket". Unless it is your birthday (encoded as a boolean value in the parameters of the function) -- on your birthday, your speed can be 5 higher in all cases *'''

def caught_speeding(speed, is_birthday):
    if is_birthday:
        speeding = speed - 5
    else:
        speeding = speed
    if speeding <= 60:
        return 'No Ticket'
    elif speeding in range(61, 81):
        return 'Small Ticket'
    else:
        return 'Big Ticket'
caught_speeding(81,True)
caught_speeding(81,False)

# Given a list of numbers and a number k, return whether any two numbers from the list add up to k.
Ex: given liss = [10, 3, 7, 15, 2, 4] and k of 25, return true since 15 + 3 + 7 is 25.Can you do this in one pass?

def list_addup(lis, k):
    for no1 in lis:
        for no2 in lis:
            if no1 + no2 == k:
                return no1, no2
            no2 += 1
        no1 += 1
    return no1, no2
list_addup(liss, 9)

def list_adds(lis, k):
    for number in lis:
        residue = k - number
        if residue in lis:
            return number, residue
list_adds(liss, 9)

def sum_exists(numbers, target):
    numbers_seen = set()
    for number in numbers:
        if target - number in numbers_seen:
            return True
        numbers_seen.add(number)
    return False
sum_exists(liss, 10)

def sum_exists1(numbers, target):
    differences = {target - number for number in numbers}
    return bool(differences.intersection(numbers))
sum_exists1(liss, 10)


{'a': [1, 2, 3, 4]}
{'a': (1, 2, 3, 4)}

def upper_to_lower(string):
    low = []
    for word in string.split(' '):
        low.append(word.lower())
    strs = ' '.join(low)
    return strs
strings = 'CREATE TABLE GAME (GAME_ID INTEGER NOT NULL PRIMARY KEY, GAME_NAME VARCHAR (20), VIDEO LONGBLOB)'
upper_to_lower(strings)

def is_divisible_by_2(numerator):
    if numerator % 2 == 0:
        return True
    else:
        return False

print(is_divisible_by_2(7))

def make_is_divisible(denominator):
    def is_divisible(numerator):
        return numerator % denominator == 0
    return is_divisible

is_div_5 = make_is_divisible(5)
is_div_2 = make_is_divisible(2)

is_div_2(15)

def num1(x):
   def num2(y):
      return x / y
   return num2
res = num1(10)
print(res(5))

no = 105
res = no/6
res1 = no//6

# Find m and c whereas y = mx+c
x1 = 0
y1 = 1
x2 = 1
y2 = 3
x3 = 2
y3 = 2
def line_eq(x, y):
    m = y / x
    c = y - m * x
    return m, c

line_eq(6, 12)

dataset = {x1: y1, x2: y2, x3: y3}

dataset.keys()
dataset[x2]

liss = [3, 5]
piss = [1, 5, 8]
ls = []
for i in range(len(liss)):
    for j in range(len(piss)):
        prods = liss[i] * piss[j]
        ls.append(prods)

def multiplys(l1, l2):
    for i in l1:
        for j in l2:
            return l1[i] * l2[j]

multiplys(l1, l2)


'''Given an array of time intervals (start, end) for classroom lectures (possibly overlapping), find min no of rooms
required
Ex: given lis = [(30, 75), (0, 50), (60, 150)], it should return 2'''
# https://hackhenry.com/daily-coding-problem-21
1. iterate thru the elements in the list
2. iterate thru the elements of the tuples in the list
3. check if 2nd element of the 1st tuple is less than or equal to 1st element of the end tuple

def overlaps(lis):
    for x in range(len(lis)):
        if lis[x][1] > :


'''A Pythagorean triplet is a set of three natural numbers, a < b < c, for which, a2 + b2 = c2
Ex: 3^2 + 4^2 = 9 + 16 = 25 = 5^2. Only one Pythagorean triplet exists for which a + b + c = 1000.Find product abc
Pythagorean triplets are in form 2m, (m^2-1), (m^2+1). Ex: 3,4,5 or 6,8,10 or 8,15,17 are triplets'''

def pyth_triplet(n):
    no = n/2
    return n, (no ** 2) - 1, (no ** 2) + 1

pyth_triplet(8)

'''
The sum of the squares of the first ten natural numbers is 1^2+2^2+...+10^2=385
The square of the sum of the first ten natural numbers is (1+2+...+10)2=55^2=3025
Difference between sum of squares of first ten natural numbers and square of the sum is 3025−385=2640.
Find difference between sum of the squares of first one hundred natural numbers and square of the sum.'''

def sumsquare_squaresum(max_no):
    tot = 0
    tot_of_sqrs = 0
    for i in range(max_no + 1):
        tot += i
        sq_tot = tot ** 2
        tot_of_sqrs += i ** 2
        difference = sq_tot - tot_of_sqrs
    return difference

sumsquare_squaresum(100)

'''
2520 is the smallest number that can be divided by each of the numbers from 1 to 10 without any remainder.
What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?'''

def lcm(n):
    ans = 1
    for i in range(1, n + 1):
        ans = int((ans * i) / math.gcd(ans, i))
    return ans

lcm(10)

data = pd.read_csv(r'C:\Users\Srees\Desktop\data\out.csv')
data.drop('Unnamed: 0', axis = 1, inplace = True)
data_json = pd.read_json(r'C:\Users\Srees\Desktop\data\roads.json')
df = pd.DataFrame(data_json) # same as above
data_json['features'][0]['geometry']['coordinates']

'''
Given two singly linked lists that intersect at some point, find the intersecting node. The lists are non-cyclical.
For example, given A = 3 -> 7 -> 8 -> 10 and B = 99 -> 1 -> 8 -> 10, return the node with value 8.
In this example, assume nodes with the same value are the exact same node objects'''


'''
The edit distance between two strings refers to the minimum number of character insertions, deletions, and
substitutions required to change one string to the other. For example, the edit distance between “kitten” and “sitting”
is 3: substitute k for s, e for i, and append a g. Compute edit distance for two strings'''
def edit_dist(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    uncomms = []
    x = set2.symmetric_difference(set1)
    return len(x)

s1 = 'kitten'
s2 = 'sitting'
edit_dist(s1, s2)

'''
Run-length encoding is a fast and simple method of encoding strings. The basic idea is to represent repeated successive
characters as a single count and character. Ex: string "AAAABBBCCDAA" encoded as "4A3B2C1D2A"'''
1. find unique chars
2. find how many times each char is repeated in string
3. concatenate no of times with string

s = 'AAAABBBCCDAA'

def encoder(strn):
    count = 0
    for i in strn:
        x = strn[0]
        if i == x:
            count += 1
            continue
        else:
            break
    return '{}{}'.format(str(count), x)

encoder(s)

def count_chars(stn):
    count = 0
    x = {}
    for i in stn:
        if i == 'A':
            count += 1
            x.__setitem__(i, count)
    return x

count_chars(s)

l = ['e', 's', 63, .77, 'sd', 'hf']
def counter(l):
    count = 0
    for i, j in enumerate(l):
        count += 1
    return count

counter(l)

# encoding
def rle_encode(data):
    encoding = ''
    prev_char = ''
    count = 1
    if not data: return ''
    for char in data:
        if char != prev_char: # If the prev and current characters don't match
            if prev_char: # then add the count and character to our encoding
                encoding += str(count) + prev_char
            count = 1
            prev_char = char
        else:
            count += 1 # Or increment our counter if the characters do match
    else:
        encoding += str(count) + prev_char # Finish off the encoding
        return encoding

encoded_val = rle_encode('AAAAAAFDDCCCCCCCAEEEEEEEEEEEEEEEEE')
print(encoded_val)

# decoding
def rle_decode(data):
    decode = ''
    count = ''
    for char in data:
        # If the character is numerical...
        if char.isdigit():
            # ...append it to our count
            count += char
        else: # Otherwise we've seen a non-numerical character and need to expand it for the decoding
            decode += char * int(count)
            count = ''
    return decode

rle_decode(encoded_val)


# first six prime numbers: 2, 3, 5, 7, 11, and 13. What is the 10 001st prime number


def new_word(lis):
    new = ''
    for i in range(len(lis)):
        if i % 2 != 0:
            new += lis[i]
    return new

l = list('Programming Tutorials')
new_word(l)

dataset = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\adult.csv')
wc = dataset.loc[::]['workclass']
stripped = wc.to_string().strip()
sers = pd.Series([stripped])
sers

string = '   This is just a test string, to play around with string functions. So lets see!   '
string.strip(' string ')

txt = ',,,,,rrttgg.....banana....rrr'
x = txt.strip(',.grt')
x

txt = "     banana     "
x = txt.strip()
x

'''use join, strip or replace to deal with unwanted characters'''

string1 = '''    geeks for geeks    '''
string1.strip()
string1.strip('[geks]')

string = '  the King has the largest army in the entire world the      '
string.replace('army i', '').strip() # works only if there is a matching pattern in string to replace

s = 'abababab'
s.replace('a', 'A', 2) # last parameter is no of times replacement should be performed

# best solution so far to remove all specified spl/chars, using string.punctuation or specified chars
line = "abc#@!?efg12;:?"
''.join( c for c in line if c not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}2~1')
string = '  the King has the largest army in the entire world the      '
''.join( c for c in string if c not in 'Kai').strip()

'''Feature Selection using Boruta'''
# Features in Boruta compete with a randomized version of themselves

# 1st implementation https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a
from boruta import BorutaPy

X = pd.DataFrame({'age': [25, 32, 47, 51, 62], 'height': [182, 176, 174, 168, 181], 'weight': [75, 71, 78, 72, 86]})
y = pd.Series([20, 32, 45, 55, 61], name = 'income')
# make X_shadow by randomly permuting each column of X
np.random.seed(42)
X_shadow = X.apply(np.random.permutation)
X_shadow.columns = ['shadow_' + feat for feat in X.columns]
# make X_boruta by appending X_shadow to X
X_boruta = pd.concat([X, X_shadow], axis = 1)

forest = RandomForestRegressor(max_depth = 5, random_state = 42) # fit a random forest (suggested max_depth between 3 and 7)
forest.fit(X_boruta, y)
# store feature importances
feat_imp_X = forest.feature_importances_[:len(X.columns)]
feat_imp_shadow = forest.feature_importances_[len(X.columns):]
# compute hits
hits = feat_imp_X > feat_imp_shadow.max()
# initialize hits counter
hits = np.zeros((len(X.columns)))
# repeat 20 times
for iter_ in range(20):
   ### make X_shadow by randomly permuting each column of X
   np.random.seed(iter_)
   X_shadow = X.apply(np.random.permutation)
   X_boruta = pd.concat([X, X_shadow], axis = 1)
   ### fit a random forest (suggested max_depth between 3 and 7)
   forest = RandomForestRegressor(max_depth = 5, random_state = 42)
   forest.fit(X_boruta, y)
   ### store feature importance
   feat_imp_X = forest.feature_importances_[:len(X.columns)]
   feat_imp_shadow = forest.feature_importances_[len(X.columns):]
   ### compute hits for this trial and add to counter
   hits += (feat_imp_X > feat_imp_shadow.max())

# probability mass function of a binomial distibution can be computed as
trials = 20
pmf = [sp.stats.binom.pmf(x, trials, .5) for x in range(trials + 1)]

boruta = BorutaPy(estimator = forest, n_estimators = 'auto', max_iter = 100) # max_iter is no of trials to perform
boruta.fit(np.array(X_boruta), np.array(y))
# print results
green_area = X_boruta.columns[boruta.support_].to_list()
blue_area = X_boruta.columns[boruta.support_weak_].to_list()


# 2nd implementation https://www.kaggle.com/rsmits/feature-selection-with-boruta
train = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\application_train.csv')
# all categorical values will be one-hot encoded
train = pd.get_dummies(train, drop_first=True, dummy_na=True)
train.shape
# get all feature names from the dataset
features = [f for f in train.columns if f not in ['TARGET','SK_ID_CURR']]
len(features)
# Replace all missing values with the Mean
train[features] = train[features].fillna(train[features].mean()).clip(-1e9, 1e9)
# get the final dataset X and labels Y
X = train[features].values
Y = train['TARGET'].values.ravel()
# setup the RandomForrestClassifier as the estimator to use for Boruta
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
# setup Boruta. With perc = 90 a threshold is specified. Lower the threshold more the features are selected
boruta_feature_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=4242, max_iter = 50, perc = 90)
boruta_feature_selector.fit(X, Y)
# After Boruta has run we can transform our dataset
X_filtered = boruta_feature_selector.transform(X)
X_filtered.shape
# create a list of the feature names to use them at a later stage
final_features = list()
indexes = np.where(boruta_feature_selector.support_ == True)
for x in np.nditer(indexes):
    final_features.append(features[x])
print(final_features)

'''Find greatest product in a diagonal'''

grid = [[8, 2, 22, 97, 38, 15, 0, 40, 0, 75, 4, 5, 7, 78, 52, 12, 50, 77, 91, 8],
			[49, 49, 99, 40, 17, 81, 18, 57, 60, 87, 17, 40, 98, 43, 69, 48, 4, 56, 62, 0],
			[81, 49, 31, 73, 55, 79, 14, 29, 93, 71, 40, 67, 53, 88, 30, 3, 49, 13, 36, 65],
			[52, 70, 95, 23, 4, 60, 11, 42, 69, 24, 68, 56, 1, 32, 56, 71, 37, 2, 36, 91],
			[22, 31, 16, 71, 51, 67, 63, 89, 41, 92, 36, 54, 22, 40, 40, 28, 66, 33, 13, 80],
			[24, 47, 32, 60, 99, 3, 45, 2, 44, 75, 33, 53, 78, 36, 84, 20, 35, 17, 12, 50],
			[32, 98, 81, 28, 64, 23, 67, 10, 26, 38, 40, 67, 59, 54, 70, 66, 18, 38, 64, 70],
			[67, 26, 20, 68, 2, 62, 12, 20, 95, 63, 94, 39, 63, 8, 40, 91, 66, 49, 94, 21],
			[24, 55, 58, 5, 66, 73, 99, 26, 97, 17, 78, 78, 96, 83, 14, 88, 34, 89, 63, 72],
			[21, 36, 23, 9, 75, 0, 76, 44, 20, 45, 35, 14, 0, 61, 33, 97, 34, 31, 33, 95],
			[78, 17, 53, 28, 22, 75, 31, 67, 15, 94, 3, 80, 4, 62, 16, 14, 9, 53, 56, 92],
			[16, 39, 5, 42, 96, 35, 31, 47, 55, 58, 88, 24, 0, 17, 54, 24, 36, 29, 85, 57],
			[86, 56, 0, 48, 35, 71, 89, 7, 5, 44, 44, 37, 44, 60, 21, 58, 51, 54, 17, 58],
			[19, 80, 81, 68, 5, 94, 47, 69, 28, 73, 92, 13, 86, 52, 17, 77, 4, 89, 55, 40],
			[4, 52, 8, 83, 97, 35, 99, 16, 7, 97, 57, 32, 16, 26, 26, 79, 33, 27, 98, 66],
			[88, 36, 68, 87, 57, 62, 20, 72, 3, 46, 33, 67, 46, 55, 12, 32, 63, 93, 53, 69],
			[4, 42, 16, 73, 38, 25, 39, 11, 24, 94, 72, 18, 8, 46, 29, 32, 40, 62, 76, 36],
			[20, 69, 36, 41, 72, 30, 23, 88, 34, 62, 99, 69, 82, 67, 59, 85, 74, 4, 36, 16],
			[20, 73, 35, 29, 78, 31, 90, 1, 74, 31, 49, 71, 48, 86, 81, 16, 23, 57, 5, 54],
			[1, 70, 54, 71, 83, 51, 54, 69, 16, 92, 33, 48, 61, 43, 52, 1, 89, 19, 67, 48]]

def largest_product(grid):
    dims = len(grid)
    prod = 0
    maxi = 0

    for i in range(0, dims):
    	for j in range(0, dims-3):
    		prod = grid[i][j] * grid[i][j+1] * grid[i][j+2] * grid[i][j+3]
    		if prod > maxi:
    			maxi = prod
    		
    for i in range(0, dims):
    	for j in range(0, dims-3):
    		prod = grid[j][i] * grid[j+1][i] * grid[j+2][i] * grid[j+3][i]
    		if prod > maxi:
    			maxi = prod
    
    for i in range(0, dims-3):
    	for j in range(0, dims-3):
    		prod = grid[j][i] * grid[j+1][i+1] * grid[j+2][i+2] * grid[j+3][i+3]
    		if prod > maxi:
    			maxi = prod
    
    for i in range(0, dims-3):
    	for j in range(3, dims):
    		prod = grid[j][i] * grid[j-1][i+1] * grid[j-2][i+2] * grid[j-3][i+3]
    		if prod > maxi:
    			maxi = prod
    return maxi

largest_product(grid)

'''''''''''''''Recursion''''''''''''
https://www.w3resource.com/python-exercises/data-structures-and-algorithms/python-recursion.php
https://www.geeksforgeeks.org/recursion-practice-problems-solutions
https://www.codesdope.com/practice/python-have-your-own-function
'''
# recursion stack limit, beyond with stack overflows
sys.getrecursionlimit()

# factorial of given input
def factorial(x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return x * factorial(x-1)
factorial(5)

# Write a recursive function to calculate the sum of numbers from 0 to n
def factorial_sum(x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return x + factorial_sum(x-1)

factorial_sum(10)

### return fibonacci nos 0, 1, 1, 2, 3, 5, 8, 13
# method 1
def no_of_fibonaccis1(no):
    fibos = [0, 1]
    start = 0
    nxt = 1
    tot = 0
    for i in range(no - 2):
        tot = start + nxt
        fibos.append(tot)
        start = nxt
        nxt = tot
    return fibos

no_of_fibonaccis1(14)

# method 2
def no_of_fibonaccis2(no):
    start = 0
    nxt = 1
    tot = 0
    print(start, nxt, end = ' ')
    for i in range(no - 2):
        tot = start + nxt
        print(tot, end = ' ')
        start = nxt
        nxt = tot

no_of_fibonaccis2(14)

# start next total
def fibonacci_upto(max_no):
    start = 0
    nxt = 1
    tot = 0
    print(start, nxt, end = ' ')
    for i in range(nxt, max_no - 1):
        tot = start + nxt
        print(tot, end = ' ')
        start = i
        i = tot

fibonacci_upto(8) # should return 0 1 1 2 3 5 8 13 21 34 55

# fibonacci sequence using recursion
def fibo(maxi):
    start = 0
    nxt = 1
    if maxi == 1:
        return start
    elif maxi == 2:
        return start, nxt
    else:
        start + nxt + fibo()

fibo(2)

# get the sum of a non-negative integer using recursion
def integer_sum(x):
    s = str(x)
    tot = 0
    for i in s:
        tot += int(i)
    return tot

integer_sum(55)

# addition upto n, method 1
def integer_sum_recursion(x):
    s = str(x)
    if len(s) == 0 | len(s) == 1:
        return x
    else:
        return int(s) + integer_sum_recursion(int(x-1))

integer_sum_recursion(55)

# addition upto n, method 2
def recursive_sum(current_no, total):
    if current_no == 11:
        return total
    else:
        return recursive_sum(current_no + 1, total + current_no)

recursive_sum(1, 0)

# same as aboove, but using global mutable state, method 3
current_number = 1
accumulated_sum = 0

def sum_recursive():
    global current_number
    global accumulated_sum
    if current_number == 11: # Base case
        return accumulated_sum
    else: # Recursive case
        accumulated_sum = accumulated_sum + current_number
        current_number = current_number + 1
        return sum_recursive()

sum_recursive()

# method 4: lisy = [1,2,3]
def recursive_list_addition(lis):
    if lis == []:
        return 0
    else:
        head = lis[0]
        nxt_list = lis[1:]
        return head + recursive_list_addition(nxt_list)

recursive_list_addition(lisy)

# write a Python program of recursion list sum. Test Data: [1, 2, [3,4], [5,6]]. Expected Result: 21
ls = [1, 2, [3,4], [5,6]]
# method 1: using iteration
def sum_up_sublists1(lis):
    tot = 0
    for i in lis:
        if type(i) == type([]):
            for j in i:
                tot += j
        else:
            tot += i
    return tot

sum_up_sublists1(ls)

# method 2: using recursion
def sum_up_sublists(lis):
    tot = 0
    for i in lis:
        if type(i) == type([]):
            tot += sum_up_sublists(i)
        else:
            tot += i
    return tot

sum_up_sublists(ls)

# find nth fibonacci no using recursion, method 1
def nth_fibo_no(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return nth_fibo_no(n-1) + nth_fibo_no(n-2)

nth_fibo_no(7)

# find nth fibonacci no using recursion, method 2
from functools import lru_cache # to cache results, avoids recomputation by explicitly checking for value before computing
@lru_cache(maxsize=None)
def nth_fibo_no2(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return nth_fibo_no(n-1) + nth_fibo_no(n-2)

nth_fibo_no2(7)

# recursive function f(n) = 3 * n, i.e. the multiples of 3
def multiples_of_x(x):

def multiples_of(x):
    if x == 1:
        return 3
    else:
        return 3 + multiples_of(x-1)

multiples_of(45)
for i in range(1, 10):
    print(multiples_of(i))

# calculate sum of positive integers of n+(n-2)+(n-4)..(until n-x =< 0)
def sum_ints(n):
    if n < 1:
        return 0
    else:
        return n + sum_ints(n - 2)

sum_ints(5)

# calculate the harmonic sum until n. harmonic sum is the sum of reciprocals of the positive integers. ex: 1+1/2+1/3+1/4..
def harmonic_sum(no): # method 1
    tot = 0
    for i in range(1, no + 1):
        tot = tot + (1 / i)
    return tot

harmonic_sum(4)

def harmonic(n): # method 2
    if n == 1:
        return 1
    else:
        return 1 / n + (harmonic(n - 1))

harmonic(4) # result = 2.0833

houses = ["Eric's house", "Kenny's house", "Kyle's house", "Stan's house"]

def recursive_home_delivery(homes):
    if len(homes) == 1:
        print('Delivering gifts to {}'.format(homes))
    else:
        mid = len(homes) // 2
        first = homes[:mid]
        nxt = homes[mid:]
        recursive_home_delivery(first)
        recursive_home_delivery(nxt)

recursive_home_delivery(houses)

# non-recursive calculate the value of 'a' to the power 'b'. ex: power(3, 4) -> 81
def power_of_base2(a, b):
    if a == 0:
        return 0
    elif b == 0:
        return 1
    elif b == 1:
        return a
    else:
        return a * power_of_base2(a, b-1)

power_of_base2(3, 4)

# find greatest common divisor (gcd) of two integers. GCD aka greatest common factor (gcf) or highest common factor (hcf)
# method 1: iterative
def gcf_iterative(n1, n2):
    factors_n1 = []
    factors_n2 = []
    common = []
    for i in range(1, n1):
        if n1 % i == 0:
            factors_n1.append(i)
    for j in range(1, n2):
        if n2 % j == 0:
            factors_n2.append(j)
    for x in factors_n1:
        if x in factors_n2:
            common.append(x)
    return max(common)

gcf_iterative(76, 52)

# method 2: recursive
def gcf_recursive(a, b):
    low = min(a, b)
    high = max(a, b)
    if low == 0:
        return high
    elif low == 1:
        return 1
    else:
        return gcf_recursive(low, high%low)

gcf_recursive(76, 52)

# print multiplication table of 12 using recursion
def table(no): # non recursive
    x = 0
    for i in range(1, 13):
        x = no * i
        print('{} * {} = {}'.format(no, i, x))

table(11)

def table_recur(no): # to do
    if no == 1:
        return 1
    else:
        return '{} * {} = {}'.format(no, table_recur(no-1), no * table_recur(no-1))

table_recur(11)
''''''
### Difference between filter() & map() functions w.r.t list comprehension.
# Map maps a lambda function, filter applies a condition to an iterable and does NOT perform an operation like map()

def sq(x):
    return x ** 2
s = [2,4,6,7]
list(map(sq, s))
list(filter(sq,s))
list(filter(lambda x: x % 2 == 0, s))

def even_check(x):
    return x % 2 == 0
list(filter(even_check, s))


# String characters balance Test: String s1 and s2 is balanced if all the chars in the s1 are there in s2
s1 = "Yn"
s2 = "PYnative" #True

s1 = "Ynf"
s2 = "PYnative" #False

def str_balance(s1, s2):
    if s1 in s2:
        return True
    else:
        return False

str_balance(s1, s2)

def str_balance2(s1, s2):
    flag = True
    for s in s1:
        if s in s2:
            continue
        else:
            flag = False
    return flag

str_balance2(s1, s2)

# Find all occurrences of “USA” in given string ignoring the case
str1 = 'Welcome to USA. usa awesome, aint it?'
def occurances(string, word):
    count = 0
    for w in string.split(' '):
        if w == 'it':
            count +=1
    return 'The {} count is: {}'.format(word, count)

occurances(str1, 'it')

# Given a string, return the sum and average of the digits that appear in the string, ignoring all other characters
str1 = 'English = 78 Science = 83 Math = 68 History = 65'
def sum_avg(string):
    nums = []
    for s in string.split(' '):
        try:
            if int(s) / 1 == int(s):
                nums.append(s)
        except:
            v = ValueError
    return nums

sum_avg(str1)

# Given an input string, count occurrences of all characters within a string
str1 = 'Apple'
{'A': 1, 'p': 2, 'l': 1, 'e': 1}
def char_occurance(string):
    c = Counter(string)
    return c

char_occurance(str1)

# method 2
def char_occurance2(string): # without inbuilt function, use set, max, list
    char = []
    repeats = []
    count = 0
    seyt = set(string) # now find repititions of these chars
    for s in seyt:
        if s in string:
            count += 1
            {s: count}
    return

char_occurance2(str1)
# Find the last position of a substring “Emma” in a given string
str1 = 'Emma is a data scientist who knows Python. Emma works at innuit'
Last occurrence of Emma starts at index 43

# Remove empty strings from a list of strings
str_list = ['Emma', 'Jon', '', 'Kelly', None, 'Eric', ''] # output: ['Emma', 'Jon', 'Kelly', 'Eric']
def empty_string_filter(st_list):
    strngs = []
    for x in st_list:
        if x == '' or x == None:
            st_list.remove(x)
    return st_list

empty_string_filter(str_list)

# Removal all the characters other than integers from string
s = 'I am 25 years and 10 months old'
output = 2510

def get_ints(st):
    ints = []
    for i in st:
        try:
            if int(i) / 1 == int(i):
                ints.append(i)
        except:
            v = ValueError
    joined = ''.join(ints)
    return int(joined)

get_ints(s)

# Find words with both alphabets and numbers
str1 = 'Emma25 is Data scientist50 and AI Expert'
Expected Output: Emma25, scientist50
def alphanumeric_words(string):
    words = []
    for s in string.split(' '):
        for i in s:
            try:
                if int(i) / 1 == int(i):
                    words.append(s)
            except:
                v = ValueError
    return words

alphanumeric_words(str1)

# Remove special symbols/Punctuation from a given string
str1 = "/*Jon is @developer & musician"
output = 'Jon is developer musician'

def remove_spl_chars(st):
    chars = []
    for s in st:
        if str(s) == str:
            chars.append(s)
    return chars

remove_spl_chars(str1)

str1.strip('/')
str1.strip('*')
str1.strip('*/')
str1.strip('@')
str1.strip('&')

import string
str1.translate(str.maketrans('', '', string.punctuation)) # or
''.join(i for i in str1 if not i in string.punctuation)

# hard-coded
def spl_chars(strng):
    schar = string.punctuation # same can be hard-coded with the known spl chars to be removed
    for s in strng.split(' '):
        for i in s:
            if i in schar:
                strng.replace(i, '')
                s.replace(i, '')
                strng.strip(i)
                s.strip(i)
    return strng
    return s

spl_chars(str1)

# use filter() for above
# use regex
# use translate

# From given string replace each punctuation with #
str1 = '/*Jon is @developer & musician!!'
Output: ##Jon is #developer # musician##


'''Visualization exercise'''
https://pynative.com/python-matplotlib-exercise
https://www.kaggle.com/jchen2186/data-visualization-with-python-seaborn

import pyforest

df = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\company_sales_data.csv')
df.sample(5)
df.columns
df.info()

# read Total profit of all months and show it using a line plot
df.plot.line(x = 'month_number', y = 'total_profit')

# get Total profit of all months and show line plot with the following Style properties
df.plot.line(x = 'month_number', y = 'total_profit', lw = 3, color = 'r', marker = 'o', linestyle = '--',
             label = 'Profit data of last year', markerfacecolor = 'black')

# Read all product sales data and show it  using a multiline plot
def lists(df):
    x = len(df.columns)
    cols = df.columns
    names = string.ascii_lowercase[0:x]
    lists = [[] for i in range(x)]
    for c in cols:
        #print({x : df[c]})
        for n in names:
            for l in lists:

lists(df)

months = df['month_number'].values.tolist()
facecream = df['facecream'].values.tolist()
facewash = df['facewash'].values.tolist()
shampoo = df['shampoo'].values.tolist()
toothpaste = df['toothpaste'].values.tolist()

import matplotlib.pyplot as plt
plt.plot(months, facecream)
plt.plot(months, toothpaste)
plt.plot(months, facewash)
plt.plot(months, shampoo)

# Read toothpaste sales data of each month and show it using a scatter plot
df.plot.scatter(x = 'month_number', y = 'toothpaste', title = 'Toothpaste sales data', label = 'Toothpaste sales data', grid = True)

# Read face cream and facewash product sales data and show it using the bar chart
df.plot(kind = 'bar', x = 'month_number', y = 'facecream')
df.plot(kind = 'bar', x = 'month_number', y = 'facewash')

# Read sales data of bathing soap of all months and show it using a bar chart. Save this plot to your hard disk
d = df.plot.bar(x = 'month_number', y = 'bathingsoap')

# Read the total profit of each month and show it using the histogram to see most common profit ranges
df.plot.hist(x = 'month_number', y = 'total_profit')

# Calculate total sale data for last year for each product and show it using a Pie chart
prods = df.columns.tolist()
df.plot.pie()

# Read all product sales data and show it using the stack plot
df['facecream'].sum()
df.plot.bar(x = df['month_number'], y = df['facecream'])

sns.barplot(x = 'month_number', y = df['facecream'], data = df)
plt.stackplot(months, facecream, facewash, shampoo, toothpaste, alpha = .7)

plt.subplots(df['bathingsoap'], df[], nrows = 2, ncols = 1, sharex= month_number, sharey)

fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (7, 7))
ax[0].plot(df['month_number'], df['bathingsoap'], linestyle = '--', color = 'r', marker = 'o', lw = 3)
ax[1].plot(df['month_number'], df['facewash'], linestyle = '--', color = 'g', marker = 'x', lw = 3)
ax[0].set_title('sales data of bathing soap')
ax[1].set_title('sales data of facewash')
ax[0].set_xlabel('months')
ax[0].set_ylabel('bathingsoap')
ax[1].set_xlabel('months')
ax[1].set_ylabel('facewash')
plt.tight_layout()

fw = sum(df['facewash'])
fc = sum(df['facecream'])
tp = sum(df['toothpaste'])
bs = sum(df['bathingsoap'])
sh = sum(df['shampoo'])
ms = sum(df['moisturizer'])

df.plot.pie(fw, fc, tp, bs, sh, ms)
labels = df.columns.tolist()
plt.pie(x = [fw, fc, tp, bs, sh, ms], explode = [0,0,0,0,0,.2], autopct = '%1.2f%%', labels = ['facewash', 'facecream', 'toothpaste',
                                                                         'bathingsoap', 'shampoo', 'moisturizer'])

class Person:

    def __init__(self, id):
        self.id = id

s = Person(100)
s.__dict__['age'] = 49
s.age + len(s.__dict__)


# automatically creating lists & its names instead of explicitly declaring
def evens_odds(lis):
    lists = [[] for i in range(5)]
    names = 'evens odds'
    listed_names = dict(zip(names.split(' '), lists))
    for x in lis:
        if x % 2 == 0:
            listed_names[names.split()[0]].append(x)
        else:
            listed_names[names.split()[1]].append(x)
    return '{} = {}, {} = {}'.format(names.split(' ')[0], listed_names[names.split(' ')[0]], names.split(' ')[1], listed_names[names.split(' ')[1]])

aList = [1, 2, 3, 4, 5, 6, 7]
evens_odds(aList)

# Concatenate two lists/tuples by index wise operation
list1 = ["M", "na", "i", "Ke"]
list2 = ["y", "me", "s", "lly"]
[x+y for x, y in zip(list1, list2)]

tup1 = ("M", "na", "i", "Ke")
tup2 = ("y", "me", "s", "lly")
[x+y for x, y in zip(tup1, tup2)]

Given a Python list. Turn every item of a list into its square
aList = [1, 2, 3, 4, 5, 6, 7]
output: [1, 4, 9, 16, 25, 36, 49]

list(map(lambda x: x**2, aList))

list1 = ["Hello ", "take "]
list2 = ["Dear", "Sir"]
output: ['Hello Dear', 'Hello Sir', 'take Dear', 'take Sir']

for i in list1:
    for j in list2:
        print(i+j)

list1 = [10, 20, 30, 40]
list2 = [100, 200, 300, 400]

i = list2.index(300)
list2[i] = 500

for i in list1:
    for j in list2[::-1]:
        print(i, j)

for x, y in zip(list1, list2[::-1]):
    print(x, y)

Remove empty strings from the list of strings
lst1 = ["Mike", "", "Emma", "Kelly", "", "Brad"]
output: ["Mike", "Emma", "Kelly", "Brad"]

def news(lis):
    for i in lis:
        if i == '':
            lis.remove(i)
    return lis

news(list1)

[x for x in lst1 if x != '']

Add item 7000 after 6000 in the following Python List
list1 = [10, 20, [300, 400, [5000, 6000], 500], 30, 40]
output: [10, 20, [300, 400, [5000, 6000, 7000], 500], 30, 40]

list1[2][2].append(7000)

list1 = ["a", "b", ["c", ["d", "e", ["f", "g"], "k"], "l"], "m", "n"]
sl = ["h", "i", "j"]

list1[2][1][2].extend(sl)

# filter words that contain atleast 2 vowels from a series
ser = ['apple', 'crypt', 'python', 'flyby', 'fork']
vowels = ['a', 'e', 'i', 'o', 'u', 'i']
vows = 'aeiou'

def vowels(lis): # to do, correct it for words having at least 2 vowels
    vow_words = []
    vows = 'aeiou'
    for i in lis:
        for j in i:
            if j in list(vows):
                vow_words.append(i)
    return list(set(vow_words))

vowels(ser)

# compare to find equality list 1 element wise squares equal to list 2
a = [11,19,11,19,19,21,6]
b = [121,361,361,121,441,121,361]
c = [121,361,121,361,361,441,36]

def equality(list1, list2):
    squares = []
    zipx = list(zip(list1, list2))
    for i1, i2 in zipx:
        if i1 ** 2 == i2:
            squares.append(i1)
    if len(squares) == len(list1) and squares == list1:
        return True
    return False

equality(a, b)

def equality2(list1, list2):
    squares = []
    for x in list1:
        y = x ** 2
        squares.append(y)
    if len(squares) == len(list1) and squares == list1:
        return True
    return False

equality2(a, b)

ampleJson = {"company":{"employee":{"name":"emma","payble":{"salary":7000,"bonus":800}}}}
gson = json.dumps(ampleJson)
gson[]

import random

def lots():
    lis = []
    for i in range(50):
        lis.append(random.randrange(1000, 9999))
    return random.sample(lis, 1)

lots()

import secrets

secret_generator = secrets.SystemRandom().randrange(1000, 9999)
secret_generator.randrange(1000, 9999)

to do
Generate a random Password which meets the following conditions:
Password length must be 10 characters long
It must contain at least 2 upper case letters, 1 digit, and 1 special symbol
hint: use random & string libraries

secrets.token_urlsafe(64)
secrets.token_hex(32)


import numpy as np

a = np.array(np.arange(1, 11)).reshape(5, 2)
a
a.shape
a.ndim

def lens(ab):
    lenss = []
    for i in ab:
        lenss.append(len(i))
    return lenss

lens(a)

np.empty(5)
ar = np.empty([4, 5], dtype = np.uint16)
ar.ndim

aray = np.array([[11 ,22, 33], [44, 55, 66], [77, 88, 99]])
aray[:, :-1]
aray[:,-1]

aray = np.array([[3 ,6, 9, 12], [15 ,18, 21, 24], [27 ,30, 33, 36], [39 ,42, 45, 48], [51 ,54, 57, 60]])
aray[::2,1::2] # odd rows & even columns

Given a two list. Create a third list by picking an odd-index element from the first list and even index elements from second
listOne = [3, 6, 9, 12, 15, 18, 21]
listTwo = [4, 8, 12, 16, 20, 24, 28]

def third(l1, l2):
    new = []
    for i in range(len(l1)):
        if i % 2 != 0:
            new.append(l1[i])
    for j in range(len(l2)):
        if j % 2 == 0:
            new.append(l2[j])
    return new

third(listOne, listTwo)

ist = [34, 54, 67, 89, 11, 43, 94]

def replace(l):
    x = l.copy()
    x.pop(4)
    print(x)
    l.insert(2, l[4])
    print(l)
    l.append(l[4])
    print(l)

replace(ist)

# given a list slice it into a 3 equal chunks and rever each list
samp = [11, 45, 8, 23, 14, 12, 78, 45, 89]

def chunks(ls):
    CHUNK_SIZE = 3
    if len(ls) % 3 == 0:
        c1 = ls[:CHUNK_SIZE] # to do also see if np.arange() with step size can be used
    return c1

chunks(samp)

def chunks2(ls):
    chunky[x:x+3]


def slice_of_3(s):
    index = 0
    lists = int(len(s) / 3)
    l = [[] for i in range(lists)]

slice_of_3(sampleList)


Given a list iterate it and count the occurrence of each element and create a dictionary to show the count of each element
original = [11, 45, 8, 11, 23, 45, 23, 45, 89]
output {11: 2, 45: 3, 8: 1, 23: 2, 89: 1}

def element_counter(ls):
    s = set(ls)
    dic = dict()
    counter = 0
    for i in range(len(ls)):
        for j in range(len(ls)-1, -1, -1):
            if ls[i] == ls[j]:
                counter += 1
                dic[i] = counter
    return dic

element_counter(original)

def repeats(s):
    count = 0
    for i in s:
        for j in s:
            if i == j:
                count += 1
        r = {i: count}
    return r

repeats('tommy lee gunnery')

def int_to_roman(num):
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syb = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syb[i]
            num -= val[i]
        i += 1
    return roman_num

int_to_roman(1100)

list(range(8-1, -1, -1))
list(range(8))

# Given a following two sets find the intersection and remove those elements from the first set
fs = {65, 42, 78, 83, 23, 57, 29}
ss = {67, 73, 43, 48, 83, 57, 29}

Intersection is {57, 83, 29}
First Set after removing common element {65, 42, 78, 23}

def inters(s1,s2):
    ls = []
    for i in s1:
        if i not in s2:
            ls.append(i)
    return set(ls)

inters(fs,ss)

checks if One Set is Subset or superset of another Set. if the subset is found delete all elements from that set
fs = {27, 43, 34}
ss = {34, 93, 22, 27, 43, 53, 48}

def subsets(l1, l2):
    for i in l1:
        if i in l2:
            return True
        return False
    for i in l2:
        if any(i) in l1:
            return True
        return False

subsets(fs, ss)

a = [1,2,3,4]
b = [1,2,3,4,5,6]

# function that accepts an array of 10 integers (between 0 and 9), that returns a string of those numbers as a phone no
ls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0] # => returns "(123) 456-7890"

def fone_no(lis):
    for i in range(len(lis)):
        

fone_no(ls)

df = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\Pokemon.csv', index_col=0, encoding= 'unicode_escape')

'''
To Do list:

OS:
rename file names

BeautifulSoup:
Scrape from portals for ranking in various categories and output as columns

Credit debit transactions
Email Slicer with Python
Create Acronyms using Python
Take user details and save to csv

take a name of user and add it to members, split it to first n last names, return email
'''


#enter 'a' to add a movie, 'l' to see movies, 'f' to find movies, 'q' to quit
movie_list = []

def movie_theatre():
    
    key = input(str('enter your choice (add/show/find/quit): '))
    while key != 'q':
        if key == 'a':
            add_movie()
        elif key == 's':
            show_movies(movie_list)
        elif key == 'f':
            find_movie()
        else:
            print('thanks for adding movies')
        key = str(input('enter a choice (add/see/find/quit): '))

def add_movie():
    
    name = str(input('Enter movie '))
    director = str(input('Enter director '))
    year = str(input('Enter released year '))
    
    movie = {'name': name, 'director': director, 'year': year}
    
    if movie not in movie_list:
        movie_list.append(movie)


def show_movies(movie_list):
    for i in movie_list:
        show_movie_details(i)


def show_movie_details(movie):
    print('Movie {}'.format(movie['name']))
    print('directed by {}'.format(movie['director']))
    print('released in {}'.format(movie['year']))


def find_movie():
    find_by = str(input('Find by: (director/film/release_year) '))
    search_for = str(input('enter search '))
    movies = find_movie_by_attribute(movie_list, search_for, lambda x: x[find_by])
    show_movies(movies)
    
    
def find_movie_by_attribute(items, expected, finder):
    found = []
    for i in items:
        if finder(i) == expected:
            found.append(i)
    return found


movie_theatre()

def adds():
    pass

a = adds
type(a)

a = adds()
type(a)


'''''''''''''''''''''''''''''Threading & multi-threading'''''''''''''''''''''''''''''
#https://www.youtube.com/watch?v=IEEhzQoKtQU

import threading
import time

start = time.perf_counter()

def do_something():
    print('sleeping 1 second')
    time.sleep(1)
    print('done sleeping')

#ex:1    
#do_something()
#do_something()

t1 = threading.Thread(target = do_something)
t2 = threading.Thread(target = do_something)

#ex:2
'''
t1.start()
t2.start()

t1.join()
t2.join()'''

#ex:3
threads = []
for _ in range(10):
    t = threading.Thread(target = do_something)
    t.start()
    thread.append(t)

for thread in threads:
    thread.join()


finish = time.perf_counter()

print('Finished in {} sec'.format(round(finish - start), 2))

a = [4,5,7,8,2,1]
for i in a:
    del(a[0])

# convert string int to int
s1 = '6485'

CHAR_DIGIT = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5,  '6':6, '7':7, '8':8, '9':9}
def str2int(num_str):
    number = 0
    for i in num_str:
        number = 10 * number + CHAR_DIGIT[i]
    return number

str2int(s1)

def add_string_nos(num_str):
    sums = 0
    for i in num_str:
        sums += CHAR_DIGIT[i]
    return sums

add_string_nos('1234')

# task 2
lis = [63,82,21,95,14]

def sorted_or_not(list_nos):
    for i in list_nos:
        if list_nos[i] < list_nos[i + 1]:
            print('list is sorted')
        else:
            print('list is not sorted')

sorted_or_not(lis)

# task 3: median of 2 arrays
a1 = [3,4,1,7]
a2 = [56,74,21,32]

#Given a list and a number k, return if any two numbers from list add up to k. ex: [10, 15, 3, 7] and k of 17, return true since 10 + 7 is 17.
n = [10, 15, 3, 7]

def sum_list(ls, no):
    for i in ls:
        for j in ls[::-1]:
            #print(i, j)
            if i + j == no:
                return True
            return False

sum_list(n, 17)
sum_list(n, 18)

class Count:
    
    def __init__(self, count = 0):
        self.count = count

c1 = Count(2)
c2 = Count(8)
id(c1) == id(c2)

s1 = 'asdf'
s2 = 'asdf'
id(s1) == id(s2)

# Return the number (count) of vowels in the given string
def vowel_counter(strng):
    vowels = ['a', 'e', 'i', 'o', 'u']
    counter = 0
    for i in strng:
        if i in vowels:
            counter += 1
    return counter
        
vowel_counter('what are you doing')

# string after processing backspace characters https://www.geeksforgeeks.org/string-after-processing-backspace-characters/
Given a string containing letters and '#', '#' represents a backspace. Print new string without '#'
Input : S = "abc#de#f#ghi#jklmn#op#"
Output : abdghjklmo
Input : S = "##geeks##for##geeks#"
Output : geefgeek

def backspace(sts):
    q = []
    for i in range(len(sts)):
        if sts[i] != '#':
            q.append(sts[i])
        elif len(q) != 0:
            q.pop()
    ans = ''.join(q) # final string
    return ans

S = "##geeks##for###geeks##"
backspace(S)

# distance converter
class Distance:

    def __init__(self, inch = None, feet = None, cms = None):
        self.inch = inch
        self.feet = feet
        self.cms = cms

    def distance_converter(self):
        self.inps = int(input('Select the below conversions: \n1. inch to feet\n2. feet to cms\n3. inch to cms\n4. Add new units\n'))
        if self.inps == 1:
            self.ins = int(input('enter inches: '))
            return inch_to_feet(self.ins)
        elif self.inps == 2:
            self.ins = int(input('enter feet: '))
            return feet_to_cms(self.ins)
        elif self.inps == 3:
            self.ins = int(input('enter inches: '))
            return inch_to_cms(self.ins)

    # supporting methods for above
    def inch_to_feet(inch):
        self.feet = self.inch / 0.08333
        return self.feet
    
    def feet_to_cms(feet):
        self.cms = self.feet / 0.03280
        return self.cms
    
    def inch_to_cms(inch):
        self.cms = self.inch * 2.54
        return self.cms

dis = Distance()
dis.distance_converter()

# allow for additional units to be added to the system
def make_function(strn, val1):
    d1 = {'kilometer_to_yard': 1093.61, 'yard_to_kilometer': 0.0009144}
    ins = input('Add your units: ')
    for i in d1.keys():
        if ins == i:
            print(d1[i])

make_function('kilometer_to_yard', 3)

d2 = {'kilometer_to_yard': 1093.61, 'yard_to_kilometer': 0.0009144'}

km_yd = 1 * 246
yd_km = 1 / 246

# Consecutive prime sum https://projecteuler.net/problem=50
Prime 41, can be written as the sum of six consecutive primes: 41 = 2 + 3 + 5 + 7 + 11 + 13

def prime(prime_no):
    # determine if a give no is prime
    primes = []
    for i in range(2, prime_no):
        for j in range(2, i):
            if i % j == 0:
                break
        else:
            primes.append(i)
    return primes

prime(50)

--> understand this else case
def ass():
    nos = []
    for i in range(10):
        for j in range(0, i):
            if i % 2 != 0:
                break
        else:
            nos.append(i)
    return nos

ass()

'''for/while else'''
for i in range(1, 4):
    print(i)
    # break # uncomment to see print statement not getting executed
else:  # Executed because no break in for
    print("No Break")

count = 0
while (count < 1):    
    count += 1
    print(count)
    break
else:
    print("No Break")

for i in range(10):
    print(i)
    if i == 9:
        print("Too big - I'm giving up!")
        break
else:
    print("Completed successfully")

# find max, 2nd max, 3rd max and min, 2nd min, 3rd min from a list of values

1. find biggest element
2. iterate thru list and calculate difference of each elements with biggest# and store in list
3. find element having smallest difference
4. return dat element

def big_small(lis):
    big = lis[0]
    diff = 0
    sec_big = lis[0]
    for i in range(len(lis)):
        if lis[i] > big:
            big = lis[i]
            i += 1
    for i in range(len(lis)):
        diff = big - lis[i]
        
        
    return big

l = [36,73,83,92,24]
big_small(l)

# try same as above by removing max element

# find first non repeating character in a string 'aabbbbcddefff' hint: try 2 use stack
ss = 'aabbbbcddefff'
def non_repeater(ss):
    letter = []
    for i in range(len(ss)):
        for j in range(i):
            print(j)

non_repeater(ss)

# find integers that add upto a given number
nos = [23,75,43,92,34,59]
def summation(lis, no):
    for i in range(lis):
        for j in range(0, i):

from functools import lru_cache# , cache # for python 3.9

for i in enumerate('kissmyass'):
    x, y = i
    z = dict(x = y)
    print(z)