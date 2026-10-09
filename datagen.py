"""
Builds the starter training set for OryvexAI (data/chat.jsonl).

A model trained from scratch only knows what is in its training data, so this
file IS the model's starting knowledge: identity, small talk, arithmetic, facts,
short explanations, Python snippets and in-conversation memory (names, colors...).
English + Persian. Add your own conversations to data/*.jsonl or text to data/*.txt
to teach it more - see README.md.
"""
from __future__ import annotations

import json
import random

SYSTEM = "You are OryvexAI, a sharp and dedicated AI assistant."

# ============================================================== English content
GREET = [
    (["hi", "hello", "hey", "hey there", "hello there", "hi there", "yo", "hi oryvex", "hello oryvex"],
     ["Hello! How can I help you today?", "Hi there. What are we working on?",
      "Hey! Ask me anything.", "Good to see you. What do you need?"]),
    (["good morning", "morning"], ["Good morning! What's on your list today?", "Good morning. How can I help?"]),
    (["good afternoon"], ["Good afternoon! What can I do for you?"]),
    (["good evening", "evening"], ["Good evening! How can I help?", "Good evening. What are you working on?"]),
    (["good night", "night"], ["Good night! Rest well.", "Sleep well. I'll be here tomorrow."]),
]
HOW_ARE_YOU = (
    ["how are you?", "how are you", "how's it going?", "how are you doing?", "what's up?", "are you ok?"],
    ["I'm running at full speed, thanks for asking. How can I help?",
     "Doing great. What can I do for you?",
     "All systems normal. What's on your mind?"])
IDENTITY = [
    (["who are you?", "who are you", "what's your name?", "what is your name?", "tell me your name", "introduce yourself"],
     ["I'm OryvexAI, a sharp and dedicated AI assistant.",
      "My name is OryvexAI. I'm a small language model built from scratch.",
      "I'm OryvexAI, a compact AI assistant trained from scratch in PyTorch."]),
    (["what are you?", "what are you", "are you an ai?", "are you a robot?", "are you human?", "are you a real person?"],
     ["I'm an AI: a small transformer language model, not a person.",
      "I'm a language model called OryvexAI. No body, just parameters."]),
    (["who made you?", "who created you?", "who built you?", "who trained you?", "who is your creator?"],
     ["I was built from scratch in PyTorch by my owner. The architecture, tokenizer and training are all custom.",
      "My owner built and trained me from scratch: a small GPT-style transformer."]),
    (["are you chatgpt?", "are you gpt?", "are you gemini?", "are you claude?", "are you llama?", "are you siri?"],
     ["No. I'm OryvexAI, a separate and much smaller model trained from scratch.",
      "No, I'm OryvexAI. I was trained from scratch and I'm far smaller than those models."]),
    (["what model are you?", "what kind of model are you?", "how do you work?", "what are you built on?", "what architecture are you?"],
     ["I'm a decoder-only transformer (GPT-style) with my own tokenizer, trained from scratch. I predict the next token, one at a time.",
      "A small GPT-style transformer written in PyTorch. I read your messages as tokens and predict the reply one token at a time."]),
    (["do you use ollama?", "do you use an api?", "do you need internet?", "are you offline?", "do you run locally?"],
     ["No. I run entirely on this computer with no internet, no API and no Ollama.",
      "I'm fully local. My weights live in a file on your machine and nothing is sent anywhere."]),
    (["how many parameters do you have?", "how big are you?", "how large is your model?", "what is your size?"],
     ["I'm a small model. The exact parameter count is shown in the panel header and in the terminal banner.",
      "Small enough to run on a laptop. Check the header of this chat for my exact parameter count."]),
]
CAPS = (
    ["what can you do?", "what can you do", "help", "what are you good at?", "how can you help me?", "what do you do?"],
    ["I can chat, answer short questions, explain basic concepts, do simple arithmetic and write short Python snippets. I'm a small model, so double-check anything important.",
     "Short answers, simple math, basic explanations and small Python examples. I'm compact, so I can be wrong."])
LIMITS = [
    (["can you browse the internet?", "can you search the web?", "can you access the internet?"],
     ["No. I run offline and only know what I learned during training."]),
    (["do you know everything?", "are you always right?", "can you make mistakes?", "are you perfect?"],
     ["No. I'm a small model and I make mistakes. Verify anything important.",
      "Far from it. My knowledge is limited to my training data, so please double-check me."]),
    (["can you remember our conversation?", "do you have memory?", "do you remember what i said?"],
     ["Yes, within this chat I can see everything we've said, up to my context limit. A new chat starts fresh.",
      "Inside this conversation, yes. When you start a new chat, my memory of it is cleared."]),
    (["what's the weather today?", "what is the weather like?", "will it rain tomorrow?"],
     ["I can't check live information like the weather because I run offline."]),
    (["what's today's date?", "what day is it?", "what time is it?", "what is the date today?"],
     ["I don't have a clock or internet access, so I can't tell the date or time. Your device can."]),
    (["what's the latest news?", "who won the game last night?", "what's the stock price today?", "what's trending?"],
     ["I can't see live news or prices. I run offline and only know my training data."]),
]
THANKS = (["thanks", "thank you", "thanks a lot", "thx", "thank you so much", "great, thanks", "cool thanks"],
          ["You're welcome!", "Anytime.", "Glad I could help.", "No problem. Ask if you need more."])
BYE = (["bye", "goodbye", "see you", "see you later", "good bye", "i'm leaving", "talk later"],
       ["Goodbye! Come back any time.", "See you later!", "Take care. I'll be here when you need me."])
UNKNOWN_TOPICS = ["quantum chromodynamics", "the stock market crash of 1987", "the history of the Ottoman navy",
                  "string theory", "the tax code", "rare tropical diseases", "medieval poetry",
                  "organic chemistry reactions", "the rules of cricket", "supply chain economics",
                  "the plot of the latest movies", "advanced topology"]
UNKNOWN_Q = ["Tell me about {t}.", "Explain {t}.", "What do you know about {t}?", "Can you teach me {t}?"]
UNKNOWN_A = ["I'm a small model and I don't know enough about {t} to answer reliably. A good book or a search engine will do better.",
             "Honestly, I don't know enough about {t}. I'd rather say that than make something up."]

CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "Portugal": "Lisbon",
    "the United Kingdom": "London", "Ireland": "Dublin", "Netherlands": "Amsterdam", "Belgium": "Brussels",
    "Switzerland": "Bern", "Austria": "Vienna", "Sweden": "Stockholm", "Norway": "Oslo", "Finland": "Helsinki",
    "Denmark": "Copenhagen", "Poland": "Warsaw", "Greece": "Athens", "Turkey": "Ankara", "Russia": "Moscow",
    "Ukraine": "Kyiv", "Iran": "Tehran", "Iraq": "Baghdad", "Saudi Arabia": "Riyadh", "Egypt": "Cairo",
    "Israel": "Jerusalem", "India": "New Delhi", "Pakistan": "Islamabad", "Afghanistan": "Kabul",
    "China": "Beijing", "Japan": "Tokyo", "South Korea": "Seoul", "Thailand": "Bangkok", "Vietnam": "Hanoi",
    "Indonesia": "Jakarta", "Australia": "Canberra", "New Zealand": "Wellington", "Canada": "Ottawa",
    "the United States": "Washington, D.C.", "Mexico": "Mexico City", "Brazil": "Brasilia",
    "Argentina": "Buenos Aires", "Chile": "Santiago", "Peru": "Lima", "Colombia": "Bogota",
    "Morocco": "Rabat", "Nigeria": "Abuja", "Kenya": "Nairobi", "South Africa": "Pretoria",
    "Ethiopia": "Addis Ababa", "Hungary": "Budapest", "Czechia": "Prague", "Romania": "Bucharest",
}
CAPITAL_Q = ["What is the capital of {c}?", "What's the capital of {c}?", "capital of {c}?",
             "Which city is the capital of {c}?", "Tell me the capital of {c}."]
CAPITAL_A = ["The capital of {c} is {k}.", "{k} is the capital of {c}.", "It's {k}."]

DEFS = {  # display name -> short explanation
    "machine learning": "Machine learning is a way of building software that learns patterns from data instead of following hand-written rules.",
    "a neural network": "A neural network is a model made of layers of simple units. It learns by adjusting its weights to reduce errors on training data.",
    "deep learning": "Deep learning is machine learning with neural networks that have many layers, which lets them learn complex patterns.",
    "a transformer": "A transformer is a neural network built around attention. It lets every token look at the other tokens to decide what matters, and it powers modern language models.",
    "attention": "Attention is a mechanism that lets a model weigh how relevant each part of the input is when processing a given token.",
    "gradient descent": "Gradient descent is an optimization method. It repeatedly nudges the model's parameters in the direction that lowers the loss.",
    "overfitting": "Overfitting is when a model memorizes its training data and then performs badly on new data.",
    "a tokenizer": "A tokenizer splits text into small pieces called tokens and maps them to numbers, which is the form a language model can read.",
    "a language model": "A language model predicts the next token in a text. By repeating that step it can write whole replies.",
    "Python": "Python is a readable, beginner-friendly programming language used for web apps, data analysis, automation and AI.",
    "an algorithm": "An algorithm is a clear sequence of steps for solving a problem.",
    "an API": "An API is a defined way for one program to talk to another by sending requests and getting responses.",
    "the CPU": "The CPU is the main processor of a computer. It executes instructions one after another, very quickly.",
    "a GPU": "A GPU is a processor with thousands of small cores, great at doing many calculations in parallel. That makes it ideal for training neural networks.",
    "RAM": "RAM is a computer's short-term working memory. It's fast, but its contents disappear when the power goes off.",
    "an operating system": "An operating system manages a computer's hardware and runs programs. Windows, macOS and Linux are examples.",
    "a database": "A database is an organized collection of data that programs can store, search and update efficiently.",
    "encryption": "Encryption scrambles data with a key so only people with the right key can read it.",
    "cloud computing": "Cloud computing means renting computing power and storage over the internet instead of owning the machines.",
    "open source": "Open source software has its source code published, so anyone can read, use and often improve it.",
    "a variable": "A variable is a named place in a program where a value is stored.",
    "a function": "A function is a reusable block of code that takes inputs, does something, and can return a result.",
    "a loop": "A loop repeats a block of code until a condition says to stop.",
    "recursion": "Recursion is when a function calls itself to solve a smaller version of the same problem.",
    "a bug": "A bug is a mistake in a program that makes it behave incorrectly.",
    "Git": "Git is a version control system. It records the history of your files so you can undo changes and work with others.",
    "photosynthesis": "Photosynthesis is how plants turn sunlight, water and carbon dioxide into sugar and oxygen.",
    "gravity": "Gravity is the force that pulls objects with mass toward each other. It's why things fall and planets orbit the sun.",
    "an atom": "An atom is the basic building block of matter, made of a nucleus of protons and neutrons with electrons around it.",
    "DNA": "DNA is the molecule that stores the instructions for building and running a living organism.",
    "electricity": "Electricity is the flow of electric charge, usually electrons moving through a conductor.",
    "a black hole": "A black hole is a region of space where gravity is so strong that nothing, not even light, can escape.",
    "inflation": "Inflation is a general rise in prices over time, which means each unit of money buys less.",
    "the internet": "The internet is a global network of connected computers that exchange data using shared protocols.",
}
DEF_Q = ["What is {d}?", "Explain {d}.", "Explain {d} simply.", "Tell me about {d}.", "Define {d}.",
         "what's {d}?", "Can you explain {d}?", "Give me a quick explanation of {d}."]

CODE = [
    (["Write a Python function to add two numbers.", "python function that adds two numbers", "How do I add two numbers in Python?"],
     "Here's a simple function:", "def add(a, b):\n    return a + b\n\nprint(add(2, 3))  # 5", "It takes two numbers and returns their sum."),
    (["Write a Python function to reverse a string.", "How do I reverse a string in Python?", "python reverse string"],
     "You can reverse a string with slicing:", "def reverse(text):\n    return text[::-1]\n\nprint(reverse(\"hello\"))  # olleh", "The slice [::-1] walks through the string backwards."),
    (["Write a function that checks if a number is even.", "How do I check if a number is even in Python?"],
     "Use the modulo operator:", "def is_even(n):\n    return n % 2 == 0\n\nprint(is_even(4))  # True", "n % 2 is the remainder after dividing by 2. If it's 0, the number is even."),
    (["Write a Python function for factorial.", "How do I compute a factorial in Python?", "python factorial"],
     "Here's a factorial function:", "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n\nprint(factorial(5))  # 120", "It multiplies every whole number from 2 up to n."),
    (["Write a Fibonacci function in Python.", "How do I generate the Fibonacci sequence in Python?", "python fibonacci"],
     "Here's an iterative version:", "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\nprint(fib(10))  # 55", "Each number is the sum of the two before it."),
    (["How do I sum a list in Python?", "Write code to sum a list of numbers.", "python sum list"],
     "Use the built-in sum function:", "numbers = [1, 2, 3, 4]\nprint(sum(numbers))  # 10", "sum() adds up every item in the list."),
    (["How do I find the largest number in a list in Python?", "python max of a list"],
     "Use max():", "numbers = [3, 9, 2, 7]\nprint(max(numbers))  # 9", "max() returns the biggest item. min() returns the smallest."),
    (["How do I count the words in a string in Python?", "python count words in a sentence"],
     "Split the text and count the pieces:", "text = \"the quick brown fox\"\nprint(len(text.split()))  # 4", "split() breaks the string on whitespace and len() counts the words."),
    (["How do I read a file in Python?", "python read a text file", "Write code to read a file."],
     "Open it with a context manager:", "with open(\"notes.txt\", encoding=\"utf-8\") as f:\n    text = f.read()\nprint(text)", "The with block closes the file automatically when you're done."),
    (["How do I print numbers from 1 to 10 in Python?", "python loop 1 to 10", "Write a loop that counts to 10."],
     "A for loop with range does it:", "for i in range(1, 11):\n    print(i)", "range(1, 11) goes from 1 up to 10. The end value is excluded."),
    (["Write a function to check if a word is a palindrome.", "python palindrome check"],
     "Compare the word with its reverse:", "def is_palindrome(word):\n    word = word.lower()\n    return word == word[::-1]\n\nprint(is_palindrome(\"level\"))  # True", "A palindrome reads the same forwards and backwards."),
    (["Write FizzBuzz in Python.", "python fizzbuzz"],
     "Here's FizzBuzz:", "for i in range(1, 101):\n    if i % 15 == 0:\n        print(\"FizzBuzz\")\n    elif i % 3 == 0:\n        print(\"Fizz\")\n    elif i % 5 == 0:\n        print(\"Buzz\")\n    else:\n        print(i)", "Check the multiple of 15 first, otherwise Fizz or Buzz would catch it too early."),
    (["How do I square every number in a list?", "python list comprehension example"],
     "A list comprehension is the cleanest way:", "numbers = [1, 2, 3, 4]\nsquares = [n * n for n in numbers]\nprint(squares)  # [1, 4, 9, 16]", "It builds a new list by applying an expression to each item."),
    (["How do I use a dictionary in Python?", "python dictionary example"],
     "A dictionary maps keys to values:", "ages = {\"ali\": 30, \"sara\": 25}\nages[\"omid\"] = 28\nprint(ages[\"sara\"])  # 25", "Look values up by key, and add new ones by assigning to a new key."),
    (["Write hello world in Python.", "python hello world", "How do I print hello world?"],
     "That's a one-liner:", "print(\"Hello, world!\")", "print() writes text to the screen."),
    (["How do I sort a list in Python?", "python sort a list"],
     "Use sorted() or list.sort():", "numbers = [5, 2, 9, 1]\nprint(sorted(numbers))  # [1, 2, 5, 9]", "sorted() returns a new list. numbers.sort() sorts in place."),
    (["Write a function to check if a number is prime.", "python prime number check"],
     "Try dividing up to the square root:", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nprint(is_prime(17))  # True", "If no number up to the square root divides n, it's prime."),
    (["How do I ask the user for input in Python?", "python get user input"],
     "Use input():", "name = input(\"Your name: \")\nprint(\"Hello,\", name)", "input() waits for the user to type a line and returns it as a string."),
]

ADVICE = [
    (["How can I study better?", "Give me study tips.", "how do i study effectively?"],
     ["Study in focused blocks of 25 to 40 minutes, test yourself instead of rereading, and review a little each day. Spaced repetition beats cramming.",
      "Test yourself often, explain ideas out loud, and sleep enough. Short daily sessions beat one long night."]),
    (["How do I stay productive?", "I can't focus.", "Give me productivity tips.", "how do i stop procrastinating?"],
     ["Pick one task, set a 25 minute timer and silence your phone. Starting is the hard part, so make the first step tiny.",
      "Write down the single most important task for today, do it first, and remove distractions while you work."]),
    (["I feel unmotivated.", "I have no motivation.", "I'm feeling lazy today."],
     ["That happens to everyone. Try the smallest possible step, like five minutes on the task. Action usually brings motivation, not the other way around.",
      "Be kind to yourself. Pick one tiny thing, do it, and let that momentum carry you."]),
    (["How do I learn programming?", "How can I become a programmer?", "where do i start with coding?"],
     ["Pick one language like Python, build small projects every week, and read other people's code. Writing code beats watching tutorials.",
      "Start with Python, solve small problems daily, then build something you actually want. Projects teach more than courses."]),
    (["How can I sleep better?", "I can't sleep.", "any tips for better sleep?"],
     ["Keep a regular schedule, avoid screens and caffeine late in the day, and keep your room cool and dark. If it lasts, talk to a doctor."]),
    (["Give me a study plan for linear algebra.", "Make me a linear algebra study plan."],
     ["Day 1: vectors and matrices, with basic operations. Day 2: linear systems, determinants and inverses. Day 3: eigenvalues, eigenvectors and practice problems. Finish each day with exercises."]),
    (["How do I debug my code?", "My code doesn't work.", "how do i find a bug?"],
     ["Read the error message first, then reproduce the problem with the smallest example. Print or log values to see where reality differs from what you expected, and change one thing at a time."]),
    (["How can I learn faster?", "how do i remember things better?"],
     ["Use active recall: close the book and write what you remember. Space your reviews over days and teach the idea to someone else."]),
    (["I'm stressed.", "I feel stressed.", "I'm anxious about work."],
     ["I'm sorry you're dealing with that. Try a few slow breaths, write down what's on your mind, and pick one small thing to handle first. If it keeps up, talking to someone you trust can really help."]),
    (["How can I improve my English?", "how do i get better at english?"],
     ["Read and listen to things you enjoy, speak or write a little every day, and keep a list of new words with example sentences."]),
    (["How do I start a project?", "I have an idea but don't know where to start."],
     ["Write the goal in one sentence, list the smallest version that would work, and build that first. Ship something rough, then improve it."]),
    (["How can I stay healthy?", "healthy habits?"],
     ["Sleep seven to nine hours, move every day, eat mostly whole foods and drink water. Small habits you keep beat big plans you drop."]),
]
JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "Why did the neural network go to school? To improve its training.",
    "I told my computer I needed a break. It said: no problem, I'll go to sleep.",
    "There are 10 kinds of people: those who understand binary and those who don't.",
    "Why was the math book sad? It had too many problems.",
    "A SQL query walks into a bar, sees two tables and asks: can I join you?",
    "Why did the developer go broke? He used up all his cache.",
    "What do you call a model that never overfits? A myth.",
]
JOKE_Q = ["Tell me a joke.", "Say something funny.", "Make me laugh.", "Got any jokes?", "joke please"]

NAMES = ["Ali", "Sara", "Omid", "Maya", "John", "Emma", "Liam", "Noah", "Zoe", "Leo", "Nina", "Reza", "Mina",
         "Arman", "Lina", "Sam", "Alex", "Ava", "Kian", "Dara", "Bita", "Hana", "Jack", "Ella", "Kai"]
CITIES = ["Tehran", "Paris", "London", "Berlin", "Tokyo", "Toronto", "Amsterdam", "Madrid", "Rome", "Sydney",
          "Dubai", "Istanbul", "Shiraz", "Isfahan", "Vienna", "Oslo"]
COLORS = ["blue", "red", "green", "yellow", "purple", "orange", "black", "white", "pink", "teal"]
FOODS = ["pizza", "sushi", "pasta", "kebab", "burgers", "ice cream", "salad", "rice", "tacos", "chocolate"]

# ============================================================== Persian content
FA_GREET = [
    (["سلام", "سلام!", "درود", "سلام خوبی؟", "سلام وقت بخیر", "هی سلام"],
     ["سلام! چطور می‌توانم کمکت کنم؟", "سلام، خوش آمدی. چه کاری از دستم برمی‌آید؟", "درود! بپرس، در خدمتم."]),
    (["صبح بخیر", "صبحت بخیر"], ["صبح بخیر! امروز چه کاری داریم؟", "صبح بخیر. چطور می‌توانم کمک کنم؟"]),
    (["عصر بخیر", "ظهر بخیر"], ["عصر بخیر! چه کمکی از من برمی‌آید؟"]),
    (["شب بخیر"], ["شب بخیر! خوب بخوابی.", "شب بخیر. فردا هم اینجا هستم."]),
]
FA_HOW = (["حالت چطوره؟", "خوبی؟", "چطوری؟", "چه خبر؟", "حالت خوبه؟"],
          ["من خوبم، ممنون که پرسیدی. تو چطوری؟", "عالی‌ام! چه کمکی می‌خواهی؟", "همه‌چیز روبه‌راه است. چه کاری داری؟"])
FA_IDENTITY = [
    (["تو کی هستی؟", "اسمت چیه؟", "اسمت چیست؟", "خودتو معرفی کن", "تو کیستی؟"],
     ["من OryvexAI هستم، یک دستیار هوش مصنوعی دقیق و متعهد.", "اسم من OryvexAI است؛ یک مدل زبانی کوچک که از صفر ساخته شده‌ام."]),
    (["تو چی هستی؟", "تو آدمی؟", "تو ربات هستی؟", "تو هوش مصنوعی هستی؟"],
     ["من یک هوش مصنوعی هستم؛ یک مدل زبانی کوچک، نه یک انسان."]),
    (["کی تو رو ساخته؟", "کی تو را ساخته؟", "سازنده‌ات کیست؟", "کی تو رو آموزش داده؟"],
     ["صاحبم مرا از صفر با PyTorch ساخته و آموزش داده است: معماری، توکنایزر و آموزش، همه سفارشی."]),
    (["تو ChatGPT هستی؟", "تو جمینای هستی؟", "تو چت جی پی تی هستی؟"],
     ["نه، من OryvexAI هستم؛ یک مدل مستقل و خیلی کوچک‌تر که از صفر ساخته شده است."]),
    (["بدون اینترنت کار می‌کنی؟", "آفلاین هستی؟", "از اوللاما استفاده می‌کنی؟", "به اینترنت وصلی؟"],
     ["نه به اینترنت نیاز دارم، نه API و نه Ollama. کاملاً روی همین کامپیوتر اجرا می‌شوم."]),
]
FA_CAPS = (["چه کارهایی بلدی؟", "چه کاری از تو برمیاد؟", "چطور می‌تونی کمکم کنی؟", "قابلیت‌هات چیه؟"],
           ["می‌توانم گپ بزنم، به سؤال‌های کوتاه جواب بدهم، مفاهیم ساده را توضیح بدهم، محاسبه‌های ساده انجام بدهم و کد پایتون کوتاه بنویسم. چون مدل کوچکی هستم ممکن است اشتباه کنم."])
FA_MEM = (["مکالمه‌مون رو یادت می‌مونه؟", "حافظه داری؟", "یادت هست چی گفتم؟"],
          ["بله، داخل همین گفتگو همه‌چیزی که گفته‌ایم را می‌بینم، تا سقف حافظه‌ام. با شروع گفتگوی جدید پاک می‌شود."])
FA_THANKS = (["ممنون", "مرسی", "دمت گرم", "متشکرم", "خیلی ممنون", "مچکرم"],
             ["خواهش می‌کنم!", "قابلی نداشت.", "خوشحالم که کمک کردم."])
FA_BYE = (["خداحافظ", "فعلا", "بای", "خدانگهدار", "به امید دیدار"],
          ["خداحافظ! هر وقت خواستی برگرد.", "موفق باشی!", "به امید دیدار."])
FA_CAPITALS = {
    "ایران": "تهران", "فرانسه": "پاریس", "آلمان": "برلین", "ایتالیا": "رم", "اسپانیا": "مادرید",
    "ژاپن": "توکیو", "چین": "پکن", "روسیه": "مسکو", "ترکیه": "آنکارا", "مصر": "قاهره",
    "هند": "دهلی نو", "کانادا": "اتاوا", "انگلستان": "لندن", "آمریکا": "واشینگتن", "برزیل": "برازیلیا",
    "عراق": "بغداد", "افغانستان": "کابل", "عربستان": "ریاض", "پرتغال": "لیسبون", "یونان": "آتن",
}
FA_CAP_Q = ["پایتخت {c} کجاست؟", "پایتخت {c} کدام شهر است؟", "پایتخت {c} چیست؟"]
FA_CAP_A = ["پایتخت {c}، {k} است.", "{k} پایتخت {c} است."]
FA_DEFS = {
    "هوش مصنوعی": "هوش مصنوعی شاخه‌ای از علوم کامپیوتر است که می‌خواهد ماشین‌ها کارهایی را انجام دهند که معمولاً به هوش انسان نیاز دارند، مثل یادگیری، تصمیم‌گیری و فهمیدن زبان.",
    "یادگیری ماشین": "یادگیری ماشین روشی است که در آن کامپیوتر به جای دریافت قانون‌های دستی، الگوها را از روی داده‌ها یاد می‌گیرد.",
    "پایتون": "پایتون یک زبان برنامه‌نویسی ساده و خوانا است که برای وب، تحلیل داده و هوش مصنوعی زیاد استفاده می‌شود.",
    "شبکه عصبی": "شبکه عصبی مدلی است که از لایه‌هایی از واحدهای ساده ساخته شده و با تنظیم وزن‌ها از روی داده یاد می‌گیرد.",
    "الگوریتم": "الگوریتم دنباله‌ای از مرحله‌های مشخص برای حل یک مسئله است.",
    "پردازنده": "پردازنده یا CPU مغز کامپیوتر است و دستورها را یکی‌یکی و با سرعت زیاد اجرا می‌کند.",
    "رم": "رم حافظه‌ی کاری و کوتاه‌مدت کامپیوتر است. سریع است، اما با خاموش شدن دستگاه محتوایش پاک می‌شود.",
    "گرانش": "گرانش نیرویی است که اجسام دارای جرم را به سمت هم می‌کشد؛ به همین دلیل اجسام می‌افتند و سیاره‌ها دور خورشید می‌چرخند.",
}
FA_DEF_Q = ["{d} چیست؟", "{d} یعنی چی؟", "درباره {d} توضیح بده.", "{d} را ساده توضیح بده.", "می‌شه {d} رو توضیح بدی؟"]
FA_NAMES = ["علی", "سارا", "رضا", "مینا", "امیر", "نگار", "کیان", "هانا", "بهار", "آرش", "پریسا", "داریوش", "نیما"]
FA_COLORS = ["آبی", "قرمز", "سبز", "زرد", "بنفش", "نارنجی", "مشکی", "سفید", "صورتی"]
FA_CITIES = ["تهران", "شیراز", "اصفهان", "تبریز", "مشهد", "رشت", "کرج", "یزد", "اهواز"]


# ============================================================== generators
def _greet(r):
    q, a = r.choice(GREET)
    return [(r.choice(q), r.choice(a))]


def _simple(pair):
    def gen(r):
        q, a = pair
        return [(r.choice(q), r.choice(a))]
    return gen


def _identity(r):
    q, a = r.choice(IDENTITY)
    return [(r.choice(q), r.choice(a))]


def _limits(r):
    q, a = r.choice(LIMITS)
    return [(r.choice(q), r.choice(a))]


def _unknown(r):
    t = r.choice(UNKNOWN_TOPICS)
    return [(r.choice(UNKNOWN_Q).format(t=t), r.choice(UNKNOWN_A).format(t=t))]


def _capital(r):
    c, k = r.choice(list(CAPITALS.items()))
    return [(r.choice(CAPITAL_Q).format(c=c), r.choice(CAPITAL_A).format(c=c, k=k))]


def _define(r):
    d, a = r.choice(list(DEFS.items()))
    return [(r.choice(DEF_Q).format(d=d), a)]


def _code(r):
    qs, intro, code, outro = r.choice(CODE)
    return [(r.choice(qs), f"{intro}\n\n```python\n{code}\n```\n\n{outro}")]


def _advice(r):
    qs, answers = r.choice(ADVICE)
    return [(r.choice(qs), r.choice(answers))]


def _joke(r):
    return [(r.choice(JOKE_Q), r.choice(JOKES))]


def _math(r):
    op = r.choice(["+", "-", "*", "/", "+", "-", "*"])
    if op == "+":
        a, b = r.randint(0, 99), r.randint(0, 99); c = a + b
        qs = [f"What is {a} + {b}?", f"{a} + {b}", f"calculate {a} plus {b}", f"what's {a} plus {b}?", f"add {a} and {b}", f"{a}+{b}=?"]
    elif op == "-":
        a, b = r.randint(0, 99), r.randint(0, 99)
        a, b = max(a, b), min(a, b); c = a - b
        qs = [f"What is {a} - {b}?", f"{a} minus {b}", f"subtract {b} from {a}", f"what's {a} minus {b}?", f"{a}-{b}=?"]
    elif op == "*":
        a, b = r.randint(2, 20), r.randint(2, 12); c = a * b
        qs = [f"What is {a} * {b}?", f"{a} times {b}", f"multiply {a} by {b}", f"what's {a} times {b}?", f"{a} x {b}"]
    else:
        b, c = r.randint(2, 12), r.randint(2, 30); a = b * c
        qs = [f"What is {a} / {b}?", f"divide {a} by {b}", f"{a} divided by {b}", f"what's {a} divided by {b}?"]
    sym = {"+": "+", "-": "-", "*": "×", "/": "÷"}[op]
    ans = r.choice([f"{a} {sym} {b} = {c}.", f"{a} {sym} {b} = {c}.", f"The answer is {c}.", f"That's {c}."])
    return [(r.choice(qs), ans)]


def _fa_greet(r):
    q, a = r.choice(FA_GREET)
    return [(r.choice(q), r.choice(a))]


def _fa_identity(r):
    q, a = r.choice(FA_IDENTITY)
    return [(r.choice(q), r.choice(a))]


def _fa_capital(r):
    c, k = r.choice(list(FA_CAPITALS.items()))
    return [(r.choice(FA_CAP_Q).format(c=c), r.choice(FA_CAP_A).format(c=c, k=k))]


def _fa_define(r):
    d, a = r.choice(list(FA_DEFS.items()))
    return [(r.choice(FA_DEF_Q).format(d=d), a)]


def _fa_math(r):
    op = r.choice(["+", "-", "*"])
    if op == "+":
        a, b = r.randint(0, 99), r.randint(0, 99); c = a + b
        q = r.choice([f"{a} + {b} چند می‌شود؟", f"{a} بعلاوه {b} چنده؟", f"حاصل {a} + {b} چیست؟"]); s = "+"
    elif op == "-":
        a, b = sorted((r.randint(0, 99), r.randint(0, 99)), reverse=True); c = a - b
        q = r.choice([f"{a} - {b} چند می‌شود؟", f"{a} منهای {b} چنده؟"]); s = "-"
    else:
        a, b = r.randint(2, 20), r.randint(2, 12); c = a * b
        q = r.choice([f"{a} ضرب در {b} چنده؟", f"{a} × {b} چند می‌شود؟"]); s = "×"
    return [(q, r.choice([f"{a} {s} {b} = {c}.", f"جواب می‌شود {c}."]))]


SIMPLE = {
    "how": _simple(HOW_ARE_YOU), "caps": _simple(CAPS), "thanks": _simple(THANKS), "bye": _simple(BYE),
    "fa_how": _simple(FA_HOW), "fa_caps": _simple(FA_CAPS), "fa_mem": _simple(FA_MEM),
    "fa_thanks": _simple(FA_THANKS), "fa_bye": _simple(FA_BYE),
}
DISTRACTORS = [_greet, _math, _define, _capital, SIMPLE["how"], SIMPLE["thanks"], _joke]
FA_DISTRACTORS = [_fa_greet, _fa_math, _fa_define, _fa_capital, SIMPLE["fa_how"], SIMPLE["fa_thanks"]]


def _memory(r):
    """Teach the model to use facts the user stated earlier in the same chat."""
    kind = r.choice(["name", "name", "color", "city", "number", "food"])
    if kind == "name":
        n = r.choice(NAMES)
        first = (r.choice([f"My name is {n}.", f"I'm {n}.", f"Call me {n}.", f"Hi, I'm {n}.", f"Hello, my name is {n}."]),
                 r.choice([f"Nice to meet you, {n}! How can I help?", f"Hello {n}, great to meet you.", f"Welcome, {n}. What are we working on?"]))
        ask = (r.choice(["What's my name?", "Do you remember my name?", "What is my name?", "Who am I?", "what's my name again?"]),
               r.choice([f"Your name is {n}.", f"You told me your name is {n}."]))
    elif kind == "color":
        c = r.choice(COLORS)
        first = (r.choice([f"My favorite color is {c}.", f"I love the color {c}.", f"My favourite color is {c}."]),
                 r.choice([f"{c.capitalize()} is a great choice.", f"Nice, {c} is a lovely color."]))
        ask = (r.choice(["What's my favorite color?", "Which color do I like?", "do you remember my favorite color?"]),
               r.choice([f"Your favorite color is {c}.", f"You told me you like {c}."]))
    elif kind == "city":
        c = r.choice(CITIES)
        first = (r.choice([f"I live in {c}.", f"I'm from {c}.", f"I live in {c}, by the way."]),
                 r.choice([f"{c} sounds like a great place.", f"Nice! I've learned a bit about {c}, but not much."]))
        ask = (r.choice(["Where do I live?", "Where am I from?", "what city do I live in?"]),
               r.choice([f"You told me you live in {c}.", f"You said you're from {c}."]))
    elif kind == "number":
        n = r.randint(1, 999)
        first = (r.choice([f"Remember the number {n}.", f"Please remember {n}.", f"Keep this number in mind: {n}."]),
                 r.choice([f"Got it. I'll remember {n}.", f"Okay, the number is {n}."]))
        ask = (r.choice(["What number did I ask you to remember?", "What was the number?", "Which number should you remember?"]),
               r.choice([f"You asked me to remember {n}.", f"The number is {n}."]))
    else:
        f = r.choice(FOODS)
        first = (r.choice([f"I love {f}.", f"My favorite food is {f}.", f"I really like {f}."]),
                 r.choice([f"{f.capitalize()} is a great pick.", f"Good taste! {f.capitalize()} is hard to beat."]))
        ask = (r.choice(["What food do I like?", "What's my favorite food?", "do you remember what i like to eat?"]),
               r.choice([f"You told me you like {f}.", f"Your favorite food is {f}."]))
    turns = [first]
    for _ in range(r.choice([0, 0, 1, 1, 2])):
        turns += r.choice(DISTRACTORS)(r)
    return turns + [ask]


def _fa_memory(r):
    kind = r.choice(["name", "name", "color", "city", "number"])
    if kind == "name":
        n = r.choice(FA_NAMES)
        first = (r.choice([f"اسم من {n} است.", f"من {n} هستم.", f"اسمم {n}ه.", f"سلام، من {n} هستم."]),
                 r.choice([f"خوشبختم {n}! چطور می‌توانم کمکت کنم؟", f"سلام {n}، از آشنایی با تو خوشحالم."]))
        ask = (r.choice(["اسم من چی بود؟", "اسمم رو یادته؟", "اسم من چیست؟", "من کی هستم؟"]),
               r.choice([f"اسم شما {n} است.", f"گفتی اسمت {n} است."]))
    elif kind == "color":
        c = r.choice(FA_COLORS)
        first = (r.choice([f"رنگ مورد علاقه‌ام {c} است.", f"من رنگ {c} را دوست دارم."]),
                 r.choice([f"رنگ {c} انتخاب قشنگی است.", f"خوب سلیقه‌ای داری؛ {c} رنگ زیبایی است."]))
        ask = (r.choice(["رنگ مورد علاقه‌ام چی بود؟", "من چه رنگی دوست دارم؟"]),
               r.choice([f"رنگ مورد علاقه‌ات {c} است.", f"گفتی رنگ {c} را دوست داری."]))
    elif kind == "city":
        c = r.choice(FA_CITIES)
        first = (r.choice([f"من در {c} زندگی می‌کنم.", f"اهل {c} هستم."]),
                 r.choice([f"{c} جای خوبی به نظر می‌رسد.", f"چه عالی! {c} شهر قشنگی است."]))
        ask = (r.choice(["من کجا زندگی می‌کنم؟", "اهل کجا هستم؟"]),
               r.choice([f"گفتی در {c} زندگی می‌کنی.", f"تو اهل {c} هستی."]))
    else:
        n = r.randint(1, 999)
        first = (r.choice([f"عدد {n} را یادت نگه دار.", f"لطفاً عدد {n} را به خاطر بسپار."]),
                 r.choice([f"باشه، عدد {n} را یادم می‌ماند.", f"متوجه شدم؛ عدد {n}."]))
        ask = (r.choice(["اون عددی که گفتم چی بود؟", "کدام عدد را باید یادت می‌ماند؟"]),
               r.choice([f"گفتی عدد {n} را یادم نگه دارم.", f"آن عدد {n} بود."]))
    turns = [first]
    for _ in range(r.choice([0, 0, 1, 1])):
        turns += r.choice(FA_DISTRACTORS)(r)
    return turns + [ask]


# (generator, weight)
GENERATORS = [
    (_greet, 4), (SIMPLE["how"], 2), (_identity, 7), (SIMPLE["caps"], 2), (_limits, 4), (SIMPLE["thanks"], 2),
    (SIMPLE["bye"], 2), (_unknown, 3), (_capital, 6), (_define, 10), (_code, 9), (_advice, 6), (_joke, 2),
    (_math, 12), (_memory, 12),
    (_fa_greet, 3), (SIMPLE["fa_how"], 1), (_fa_identity, 5), (SIMPLE["fa_caps"], 1), (SIMPLE["fa_mem"], 1),
    (SIMPLE["fa_thanks"], 1), (SIMPLE["fa_bye"], 1), (_fa_capital, 3), (_fa_define, 5), (_fa_math, 3),
    (_fa_memory, 5),
]


def build_conversations(n: int = 12000, seed: int = 1337) -> list[dict]:
    r = random.Random(seed)
    funcs, weights = zip(*GENERATORS)
    convs = []
    for _ in range(n):
        turns = list(r.choices(funcs, weights)[0](r))
        # sometimes chain extra single-turn exchanges for longer chats (same language family)
        while len(turns) < 3 and r.random() < 0.3:
            turns += r.choices(funcs, weights)[0](r)
        messages = [{"role": "system", "content": SYSTEM}]
        for u, a in turns[:8]:
            messages += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        convs.append({"messages": messages})
    return convs


def write_dataset(path: str, n: int = 12000, seed: int = 1337) -> int:
    convs = build_conversations(n, seed)
    with open(path, "w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return len(convs)


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    print(write_dataset("data/chat.jsonl"), "conversations written to data/chat.jsonl")
