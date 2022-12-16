# @desc 
# @author frankliu
# @time 2022
if __name__ == '__main__':
    n = int(input())
    if n < 1 or n > 200:
        print('Error')
    else:
        res = 1
        for i in range(1, n + 1):
            res = res * i
        print(res)
