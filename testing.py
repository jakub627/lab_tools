import lab_tools as lt


def main():
    v1 = lt.Variable(10, 1)
    v2 = lt.Variable(20, 1)
    print((v1 + v2).n)


if __name__ == "__main__":
    main()
