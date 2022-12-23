case1 = "10"

# negative integer as string

case2 = "-10"

# float as string

case3 = "19.90"

# a string

case4 = "Favtutor article"

# an integer and a string

case5 = "10 Methods"

# an integer

case6 = 10

# creating a list of all testcases

testcases = [case1, case2, case3, case4, case5, case6]

# creating for loop for the testcases list

for case in testcases:

    # calling the isdigit() method

    if case.isdigit():

        print(case, " is an integer.")

    else:

        print(case, " is not an integer.")
