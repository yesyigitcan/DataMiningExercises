import pandas
class Rule:
    def __init__(self, A, B=None):
        # {A} -> {B}
        if type(A) == list:
            self.A = tuple(A)
        else:
            self.A = A
        if type(B) == list:
            self.B = tuple(B)
        else:
            self.B = B

    def text(self):
        return self.__str__()
    def __str__(self):
        aText = ""
        bText = ""
        if type(self.A) in (list, tuple):
            for element in self.A:
                aText += str(element) + ", "
            if aText != "":
                aText = aText[:-2]
        else:
            aText = str(self.A)

        if self.B:
            if type(self.B) in (list, tuple):
                for element in self.B:
                    bText += str(element) + ", "
                if bText != "":
                    bText = bText[:-2]
            else:
                bText = str(self.B)

            return "{" + aText + "} -> {" + bText + "}" 
        else:
            return "{" + aText + "}"

    

class association:
    @staticmethod
    def support(df, rule):
        if type(rule) != Rule:
            raise Exception("This method only works with Rule objects")
        if not rule.A:
            raise Exception("There must be some attributes in left side of the rule")

        length = float(len(df))
        if type(rule.A) in (list, tuple):
            searchCountA = len(df[df.apply(lambda x: search.isIn(list(x), rule.A), axis=1) == True])
        else:
            searchCountA = len(df[df.apply(lambda x: search.isIn(list(x), [rule.A]), axis=1) == True])
        if rule.B:
            if type(rule.B) in (list, tuple):
                searchItems = list(rule.B)
            else:
                searchItems = [rule.B]
            if type(rule.A) in (list, tuple):
                for att in rule.A:
                    if att not in searchItems:
                        searchItems.append(att)
            else:
                if rule.A not in searchItems:
                    searchItems.append(rule.A)

            searchCountB = len(df[df.apply(lambda x: search.isIn(list(x), searchItems), axis=1) == True])
            return searchCountB / searchCountA
        else:
            return searchCountA / length
        

    @staticmethod
    def support_value(df, rules):
        if type(rules) in (tuple, list):
            if type(rules[0]) != Rule:
                raise Exception("This method only works with Rule objects")
        else:
            raise Exception("Your collection must be in type of list or tuple")

        output = list()
        for rule in rules:
            if rule.B:
                if type(rule.A) in (list, tuple):
                    df_temp = df[df.apply(lambda x: search.isIn(list(x), rule.A), axis=1) == True]
                else:
                    df_temp = df[df.apply(lambda x: search.isIn(list(x), [rule.A]), axis=1) == True]
                length = float(len(df_temp))
                if type(rule.B) in (list, tuple):
                    searchCount = len(df_temp[df_temp.apply(lambda x: search.isIn(list(x), rule.B), axis=1) == True])
                else:
                    searchCount = len(df_temp[df_temp.apply(lambda x: search.isIn(list(x), [rule.B]), axis=1) == True])

                if length == 0 and searchCount != 0:
                    raise Exception("Search count is not zero although length is zero")
                elif length == 0:
                    output.append(0)
                else:
                    output.append(searchCount/length)

            else:
                length = float(len(df))
                if type(rule.A) in (list, tuple):
                    searchCount = len(df[df.apply(lambda x: search.isIn(list(x), rule.A), axis=1) == True])
                else:
                    searchCount = len(df[df.apply(lambda x: search.isIn(list(x), [rule.A]), axis=1) == True])
                output.append(searchCount/length)
        return tuple(output)

    @staticmethod
    def confidence_value(df, rules):
        if type(rules) in (tuple, list):
            if type(rules[0]) != Rule:
                raise Exception("This method only works with Rule objects")
        else:
            raise Exception("Your collection must be in type of list or tuple")

        output = list()
        for rule in rules:
            if rule.B:
                if type(rule.A) in (list, tuple):
                    df_temp = df[df.apply(lambda x: search.isIn(list(x), rule.A), axis=1) == True]
                else:
                    df_temp = df[df.apply(lambda x: search.isIn(list(x), [rule.A]), axis=1) == True]
                length = float(len(df_temp))

                
                if(type(rule.B) in (list, tuple)):
                    searchItems = list(rule.B)
                else:
                    searchItems = [rule.B]
                
                if(type(rule.A) in (list, tuple)):
                    for att in rule.A:
                        if att not in searchItems:
                            searchItems.append(att)
                else:
                    if rule.A not in searchItems:
                        searchItems.append(rule.A)

                searchCount = len(df_temp[df_temp.apply(lambda x: search.isIn(list(x), searchItems), axis=1) == True])

                if length == 0 and searchCount != 0:
                    raise Exception("Search count is not zero although length is zero")
                elif length == 0:
                    output.append(0)
                else:
                    output.append(searchCount/length)

            else:
                raise Exception("There must be some attribute on the right side of the rule in order to calculate confidence value")
        return tuple(output)

    @staticmethod
    def lift_value(df, rules):
        output = list()
        for rule in rules:
            if not rule.A:
                raise Exception("There must be some attributes on the left side of a rule")
            if not rule.B:
                raise Exception("There must be some attributes on the right side of a rule for lift value")
            if type(rule.B) in (list, tuple):
                searchItems = list(rule.B)
            else:
                searchItems = [rule.B]
            if type(rule.A) in (list, tuple):
                for att in rule.A:
                    if att not in searchItems:
                        searchItems.append(att)
            else:
                if rule.A not in searchItems:
                    searchItems.append(rule.A)
            supportA = association.support(df, Rule(rule.A))
            supportB = association.support(df, Rule(rule.B))
            supportConcatenate = association.support(df, Rule(searchItems))

            value = supportConcatenate / (supportA * supportB)
            output.append(value)
        return tuple(output)



class search:
    @staticmethod
    def isIn(collection, searchItems):
        if type(searchItems) not in (tuple, list):
            raise Exception("Search items should be stored in list or tuple")
        for element in searchItems:
            if element not in collection:
                return False
        return True




if __name__ == '__main__':
    df = pandas.read_csv('market_sales.csv').drop("Item_Count", axis=1).astype('str')
    
    '''
    # Delete the comment signs to see unique item names in all columns of the table
    uniqueItemNames = []
    for key in df.keys():
        for itemName in list(df[key].unique()):
            if itemName not in uniqueItemNames:
                uniqueItemNames.append(itemName)

    for count, itemName in enumerate(sorted(uniqueItemNames)):
        print(count, itemName)
    '''

    # Here I combine some items by some keywords. ex. citrus fruit, tropical fruit, .. -> fruit
    # "canned fruit", "citrus fruit", "frozen fruits", "packaged fruit/vegetables", "tropical fruit"
    # I assume fruit/vegetable juice also as in fruit category since it is made of fruit and related to fruit
    for key in df.keys():
        for itemName in list(df[key].unique()):
            if "fruit" in itemName:
                df[key] = df[key].replace(itemName, 'fruit')
            elif "beer" in itemName:
                df[key] = df[key].replace(itemName, 'beer')

    rules = (
        Rule("whole milk"),
        Rule("yogurt"),
        Rule("coffee"),
        Rule("fruit"),
        Rule("sugar"),
        Rule("hamburger meat"),
        Rule("ketchup"),
        Rule("soda"),
        Rule("chicken"),
        Rule("pork")
    )

    '''
    print("SUPPORT VALUE FOR ONE SIDED RULE")
    sup1 = association.support_value(df, rules)
    for i in range(len(rules)):
        print(rules[i], sup1[i])
    '''
    

    rules = (
        Rule("whole milk", "yogurt"),
        Rule("other vegetables", "whole milk"),
        Rule("coffee", "fruit"),
        Rule("coffee", "sugar"),
        Rule("soda", "coffee"),
        Rule("hamburger meat", "ketchup"),
        Rule(("whole milk", "yogurt"), "coffee"),
        Rule(("coffee", "soda"), "beer"),
        Rule(("chicken", "pork"), "beef"),
        Rule(("chicken", "pork", "beef"), "other vegetables")
    )

    rules_reversed = (
        Rule("yogurt", "whole milk"),
        Rule("whole milk", "other vegetables"),
        Rule("fruit", "coffee"),
        Rule("sugar", "coffee"),
        Rule("coffee", "soda"),
        Rule("ketchup", "hamburger meat"),
        Rule("coffee", ("whole milk", "yogurt")),
        Rule("beer", ("coffee", "soda")),
        Rule("beef", ("chicken", "pork")),
        Rule("other vegetables", ("chicken", "pork", "beef"))
    )


    '''
    print("SUPPORT VALUE FOR TWO SIDED RULE")
    sup2 = association.support_value(df, rules)
    for i in range(len(rules)):
        print(rules[i], sup2[i])
    '''

    '''
    print("CONFIDENCE VALUE")
    conf = association.confidence_value(df, rules)
    conf_reversed = association.confidence_value(df, rules_reversed)
    for i in range(len(rules)):
        print(rules[i], conf[i])

    print("\n")

    print("CONFIDENCE VALUE FOR SAME RULES BUT REVERSED")
    for i in range(len(rules_reversed)):
        print(rules_reversed[i], conf_reversed[i])
    '''

    '''
    print("LIFT VALUE")
    lift = association.lift_value(df, rules)
    for i in range(len(rules)):
        print(rules[i], lift[i])
    '''