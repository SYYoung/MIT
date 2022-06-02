s = "azcbobobegghakl"
s = "abcbcd"
s = ""

max_start, max_end = 0,0
max_len = 0


cur_start, cur_end = 0,0
for i in range(len(s)-2):
    if s[i] <= s[i+1] :
        cur_end = i + 1
        cur_len = cur_end - cur_start + 1
    else:
        if cur_len > max_len:
            max_start = cur_start
            max_end = cur_end
            max_len = cur_len
        cur_start = i + 1

print("The longest string is: " + s[max_start:max_end+1])

## Q3
## initialization
print("Please think of a number between 0 and 100!")
bingo = False
low, high = 0, 100
ans = (low + high)//2
prompt1 = "Enter 'h' to indicate the guess is too high. "
prompt2 = "Enter 'l' to indicate the guess is too low. "
prompt3 = "Enter 'c' to indicate I guessed correctly. "
prompt = prompt1 + prompt2 + prompt3
accept_input = ['c', 'h', 'l']

while not bingo:
    ans = (low + high)//2
    print("Is your secret number number " + str(ans) +"?")
    reply = input(prompt)
    while (reply not in accept_input):
        print("Sorry, I did not understand your input.")
        print("Is your secret number " + str(ans) + "?")
        reply = input(prompt)
    if (reply == 'c'):
        bingo = True
        print("Game over. Your secret number was: " + str(ans))
    elif (reply == 'h'):
        high = ans
    elif (reply == 'l'):
        low = ans
