from lab_tools.unc.core import ufloat

# from uncertainties import ufloat


def main():
    v1 = ufloat(2, 7)
    v2 = ufloat(1, 3)
    v3 = ufloat(1, (75) ** 0.5)
    print(f"{v1**-2:f}")


if __name__ == "__main__":
    main()
