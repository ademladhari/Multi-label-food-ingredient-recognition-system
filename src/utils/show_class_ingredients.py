# scripts/show_class_ingredients.py
import os
cls = "Aburaage"
meta = os.path.join("data","metadata","class_ingredient.txt")
ing = os.path.join("data","metadata","ingredient_list_label.txt")

with open(meta, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        if line.startswith(cls+" "):
            parts = line.strip().split()
            # first token is class
            bits = parts[1:]
            ones = [i for i,v in enumerate(bits) if v=='1']
            print(f"Class {cls} has {len(ones)} positive ingredient indices")
            # load ingredient names
            names = []
            with open(ing, encoding='utf-8') as f2:
                for L in f2:
                    L=L.strip()
                    if not L:
                        continue
                    if ',' in L:
                        _,n = L.split(',',1)
                        names.append(n.strip())
                    else:
                        names.append(L)
            for idx in ones:
                name = names[idx] if idx < len(names) else '<unknown>'
                print(idx, name)
            break
    else:
        print(f"Class {cls} not found in {meta}")
