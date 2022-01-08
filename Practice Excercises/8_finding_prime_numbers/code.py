

for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:  # if n is divisible by x, it means it's not a prime number.
            print(f"{n} equals {x} * {n//x}")
            break
    else:  # if n was not divisible by any x, it means it is a prime number.
        print(f"{n} is a prime number.")







n = int(input("Enter Number-> "))

chk=0

for i in range(2,n):
    if(n%i == 0):
        print("Number is not prime")
        chk=1
        break;
        
if(chk==0):
    print("Number is Prime")