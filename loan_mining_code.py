# Khai phá luật kết hợp & phân lớp hồ sơ vay


# 0. Cài đặt thư viện & tải dữ liệu
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "mlxtend", "gradio"])
from google.colab import files
uploaded = files.upload()
import warnings
try:
    from IPython.display import display
except ImportError:
    display = print
warnings.filterwarnings("ignore")
import time, sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, fbeta_score, precision_score,
                             recall_score, f1_score, accuracy_score)

MIN_SUPPORT = 0.02
MIN_CF = 0.30
MAX_ITEMS = 3

pd.set_option("display.max_colwidth", None)

POLICY_MIN_INCOME = 25_000
POLICY_MAX_LPI = 0.4
MAX_VIOLATION = 0.10


# 1. Đọc & làm sạch dữ liệu
df = pd.read_csv("loan_data.csv")
print("Ban đầu:", df.shape)
df = df[(df.person_age <= 80) & (df.person_emp_exp <= 60)].drop_duplicates().reset_index(drop=True)
df["loan_status"] = df["loan_status"].map({1: "Approved", 0: "Rejected"})
print("Sau làm sạch:", df.shape)
print(df["loan_status"].value_counts(normalize=True).round(3))


# 2. EDA
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for status, color in [("Approved", "tab:green"), ("Rejected", "tab:red")]:
    sub = df[df.loan_status == status]
    ax[0].hist(sub.credit_score, bins=30, alpha=0.6, label=status, color=color)
    ax[1].hist(sub.loan_percent_income, bins=30, alpha=0.6, label=status, color=color)
ax[0].set_title("credit_score"); ax[0].legend()
ax[1].set_title("loan_percent_income"); ax[1].legend()
pd.crosstab(df.previous_loan_defaults_on_file, df.loan_status, normalize="index").mul(100).plot(
    kind="bar", ax=ax[2], color=["tab:green", "tab:red"], rot=0)
ax[2].set_title("% duyệt theo lịch sử vỡ nợ")
plt.tight_layout(); plt.show()


# 3. Rời rạc hoá & kiểm tra đa cộng tuyến


# 3a. Tương quan giữa các biến số
num_cols = ["person_age", "person_income", "person_emp_exp", "loan_amnt", "loan_int_rate",
            "loan_percent_income", "cb_person_cred_hist_length", "credit_score"]
corr = df[num_cols].corr()
print("Cặp tương quan cao (|r| >= 0.8):")
for i, a in enumerate(num_cols):
    for b in num_cols[i + 1:]:
        if abs(corr.loc[a, b]) >= 0.8:
            print(f"   {a} ~ {b}: r = {corr.loc[a, b]:.2f}")
plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1); plt.colorbar()
plt.xticks(range(8), num_cols, rotation=90); plt.yticks(range(8), num_cols)
plt.title("Tương quan Pearson"); plt.show()


# 3b. Rời rạc hoá bằng ngưỡng nghiệp vụ
BINS = {
    "credit_score":        ([0, 580, 670, 740, np.inf],           ["Poor", "Fair", "Good", "VeryGood"]),
    "person_income":       ([0, 30_000, 60_000, 100_000, np.inf], ["<30k", "30-60k", "60-100k", ">=100k"]),
    "loan_amnt":           ([0, 5_000, 10_000, 20_000, np.inf],   ["<5k", "5-10k", "10-20k", ">=20k"]),
    "loan_int_rate":       ([0, 10, 13, 16, np.inf],              ["<10%", "10-13%", "13-16%", ">=16%"]),
    "loan_percent_income": ([0, 0.10, 0.20, 0.30, np.inf],        ["<10%", "10-20%", "20-30%", ">=30%"]),
    "person_age":          ([0, 25, 35, np.inf],                  ["<25", "25-34", ">=35"]),
}
for col, (edges, labels) in BINS.items():
    df[col + "_band"] = pd.cut(df[col], bins=edges, labels=labels, right=False).astype(str)

ATTRS = [c + "_band" for c in BINS] + ["person_gender", "person_education", "person_home_ownership",
                                        "loan_intent", "previous_loan_defaults_on_file"]
print(len(ATTRS), "thuộc tính:", ATTRS)
df["credit_score_band"].value_counts()


# 3c. Cramér's V giữa các thuộc tính rời rạc
def cramers_v(x, y):
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table, correction=False)[0]
    return np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))

V = pd.DataFrame([[cramers_v(df[a], df[b]) for b in ATTRS] for a in ATTRS], index=ATTRS, columns=ATTRS)
high = [(a, b, round(V.loc[a, b], 2)) for i, a in enumerate(ATTRS) for b in ATTRS[i + 1:] if V.loc[a, b] >= 0.5]
print("Cặp V >= 0.5:", high if high else "không có")
plt.figure(figsize=(8, 7)); plt.imshow(V, cmap="Oranges", vmin=0, vmax=1); plt.colorbar()
plt.xticks(range(len(ATTRS)), ATTRS, rotation=90); plt.yticks(range(len(ATTRS)), ATTRS)
plt.title("Cramér's V"); plt.show()


# 4. Chia dữ liệu train / validation / test (60/20/20)
train, test = train_test_split(df, test_size=0.2, stratify=df.loan_status, random_state=42)
train, val = train_test_split(train, test_size=0.25, stratify=train.loan_status, random_state=42)
print(len(train), len(val), len(test))


# 5. Data Warehouse (star schema trên SQLite)
conn = sqlite3.connect("loan_dw.sqlite")
DIMS = {"education": "person_education", "home": "person_home_ownership", "intent": "loan_intent",
        "gender": "person_gender", "prev_default": "previous_loan_defaults_on_file", "status": "loan_status"}
fact = df[["person_income", "loan_amnt", "loan_int_rate", "loan_percent_income", "credit_score",
           "person_age"] + [c + "_band" for c in BINS]].copy()
for name, col in DIMS.items():
    codes, values = pd.factorize(df[col], sort=True)
    pd.DataFrame({"id": range(len(values)), "value": values}).to_sql(f"dim_{name}", conn, index=False, if_exists="replace")
    fact[f"{name}_id"] = codes
fact.to_sql("fact_loan", conn, index=False, if_exists="replace")

select = ", ".join(f"dim_{n}.value AS {c}" for n, c in DIMS.items())
joins = " ".join(f"JOIN dim_{n} ON fact_loan.{n}_id = dim_{n}.id" for n in DIMS)
conn.execute("DROP VIEW IF EXISTS v_loan"); conn.execute("DROP VIEW IF EXISTS v_policy")
conn.execute(f"CREATE VIEW v_loan AS SELECT fact_loan.*, {select} FROM fact_loan {joins}")
conn.execute(f'''CREATE VIEW v_policy AS SELECT *,
    (person_income < {POLICY_MIN_INCOME} OR loan_percent_income > {POLICY_MAX_LPI}) AS violation
    FROM v_loan''')

def sql(query):
    return pd.read_sql_query(query, conn)

PCT = "ROUND(100.0 * AVG(loan_status = 'Approved'), 1)"
print(sql("SELECT COUNT(*) AS so_dong FROM v_loan"))


# 5a. OLAP — Slice: tỉ lệ duyệt theo lịch sử vỡ nợ
q = sql(f"SELECT previous_loan_defaults_on_file AS prev_default, COUNT(*) AS n, {PCT} AS pct_approved "
        "FROM v_loan GROUP BY prev_default")
print(q)
q.plot(x="prev_default", y="pct_approved", kind="bar", rot=0, legend=False, title="% duyệt theo lịch sử vỡ nợ")
plt.show()


# 5b. OLAP — ROLLUP(học vấn → sở hữu nhà)
q = sql('''
  SELECT person_education AS edu, person_home_ownership AS home, COUNT(*) AS n, ROUND(AVG(loan_amnt)) AS avg_loan
  FROM v_loan GROUP BY edu, home
  UNION ALL SELECT person_education, 'ALL', COUNT(*), ROUND(AVG(loan_amnt)) FROM v_loan GROUP BY person_education
  UNION ALL SELECT 'ALL', 'ALL', COUNT(*), ROUND(AVG(loan_amnt)) FROM v_loan''')
print(q.to_string())
q[(q.edu != "ALL") & (q.home != "ALL")].pivot(index="edu", columns="home", values="avg_loan").plot(
    kind="bar", rot=0, title="ROLLUP: khoản vay bình quân theo học vấn × sở hữu nhà")
plt.show()


# 5c. OLAP — CUBE(mục đích vay, giới tính): mọi tổ hợp con (intent,gender), (intent), (gender), ()
q = sql(f'''
  SELECT loan_intent AS intent, person_gender AS gender, {PCT} AS pct FROM v_loan GROUP BY intent, gender
  UNION ALL SELECT loan_intent, 'ALL', {PCT} FROM v_loan GROUP BY loan_intent
  UNION ALL SELECT 'ALL', person_gender, {PCT} FROM v_loan GROUP BY person_gender
  UNION ALL SELECT 'ALL', 'ALL', {PCT} FROM v_loan''')
cube = q.pivot(index="intent", columns="gender", values="pct")
print(cube)
plt.imshow(cube, cmap="RdYlGn"); plt.colorbar(label="% duyệt")
plt.xticks(range(cube.shape[1]), cube.columns); plt.yticks(range(cube.shape[0]), cube.index)
plt.title("CUBE: % duyệt theo mục đích vay × giới tính"); plt.show()


# 5d. Truy vấn thực tế 1 — Quản lý danh mục: dư nợ đã duyệt tập trung vào mục đích vay nào?
q = sql('''SELECT loan_intent, COUNT(*) AS so_ho_so, SUM(loan_amnt) AS du_no FROM v_loan
           WHERE loan_status = 'Approved' GROUP BY loan_intent ORDER BY du_no DESC''')
q["ty_trong_%"] = (100 * q.du_no / q.du_no.sum()).round(1)
print(q)
q.plot(x="loan_intent", y="du_no", kind="bar", rot=30, legend=False, title="Dư nợ đã duyệt theo mục đích vay")
plt.show()


# 5e. Truy vấn thực tế 2 — Phòng rủi ro: ma trận % duyệt theo điểm tín dụng × lịch sử vỡ nợ
q = sql(f'''SELECT credit_score_band, previous_loan_defaults_on_file AS prev_default, COUNT(*) AS n, {PCT} AS pct
            FROM v_loan GROUP BY credit_score_band, prev_default''')
m = q.pivot(index="credit_score_band", columns="prev_default", values="pct").loc[["Poor", "Fair", "Good", "VeryGood"]]
print(m)
m.plot(kind="bar", rot=0, title="% duyệt theo điểm tín dụng × lịch sử vỡ nợ"); plt.show()


# 5f. Truy vấn thực tế 3 — Phòng chính sách: 5 tổ hợp (mục đích, nhà) có n ≥ 500 mà tỉ lệ duyệt thấp nhất
q = sql(f'''SELECT loan_intent || ' / ' || person_home_ownership AS nhom, COUNT(*) AS n, {PCT} AS pct
            FROM v_loan GROUP BY loan_intent, person_home_ownership HAVING n >= 500 ORDER BY pct LIMIT 5''')
print(q)
q.plot(x="nhom", y="pct", kind="barh", legend=False, title="5 nhóm có % duyệt thấp nhất"); plt.show()


# 5g. Truy vấn thực tế 4 — Kiểm soát nội bộ: hồ sơ vi phạm năng lực trả nợ có đang được duyệt không?
q = sql(f'''SELECT credit_score_band, violation, COUNT(*) AS n, {PCT} AS pct FROM v_policy
            WHERE previous_loan_defaults_on_file = 'No' GROUP BY credit_score_band, violation''')
m = q.pivot(index="credit_score_band", columns="violation", values="pct").loc[["Poor", "Fair", "Good", "VeryGood"]]
m.columns = ["Không vi phạm", "Vi phạm"]
print(m)
m.plot(kind="bar", rot=0, title="% duyệt: vi phạm vs không vi phạm (nhóm chưa từng vỡ nợ)"); plt.show()


# 6. Khai phá luật kết hợp: Apriori vs FP-Growth
def to_items(data):
    return pd.get_dummies(data[ATTRS].astype(str), prefix_sep="=").astype(bool)

X_tr = to_items(train)
ITEMS = list(X_tr.columns)
X_val = to_items(val).reindex(columns=ITEMS, fill_value=False)
X_te = to_items(test).reindex(columns=ITEMS, fill_value=False)
y_tr, y_val, y_te = train.loan_status.values, val.loan_status.values, test.loan_status.values

basket = X_tr.copy()
basket["class=Approved"] = y_tr == "Approved"
basket["class=Rejected"] = y_tr == "Rejected"

results = {}
for name, algo, extra in [("Apriori", apriori, {"low_memory": True}), ("FP-Growth", fpgrowth, {})]:
    t = time.time()
    itemsets = algo(basket, min_support=MIN_SUPPORT, use_colnames=True, max_len=MAX_ITEMS + 1, **extra)
    results[name] = (itemsets, time.time() - t)
    print(f"{name:<10} {len(itemsets):>6} itemset, {results[name][1]:.2f} giây")
same = set(results["Apriori"][0].itemsets) == set(results["FP-Growth"][0].itemsets)
print("Hai thuật toán cho cùng tập itemset:", same)
print("-> Dùng itemset của Apriori cho mọi bước sau: kết quả giống hệt FP-Growth, chạy nhanh hơn trên dữ liệu này.")
r = association_rules(results["Apriori"][0], metric="confidence", min_threshold=0)
is_class = lambda s: len(s) == 1 and next(iter(s)).startswith("class=")
r = r[r.consequents.apply(is_class) & ~r.antecedents.apply(lambda s: any(i.startswith("class=") for i in s))]

rules = pd.DataFrame({
    "ante": r.antecedents.values,
    "cls": r.consequents.apply(lambda s: next(iter(s))[6:]).values,
    "support": r.support.values, "confidence": r.confidence.values, "lift": r.lift.values})
rules["n_items"] = rules.ante.apply(len)
prior = pd.Series(y_tr).value_counts(normalize=True)
rules["cf"] = (rules.confidence - rules.cls.map(prior)) / (1 - rules.cls.map(prior))
rules = rules[rules.cf >= MIN_CF].reset_index(drop=True)

print(rules.cls.value_counts())
print("\nVí dụ minh hoạ — 5 luật lift cao nhất mỗi lớp:")
for cls in ["Approved", "Rejected"]:
    for _, x in rules[rules.cls == cls].nlargest(5, "lift").iterrows():
        print(f"  {' ∧ '.join(sorted(x.ante))} → {cls} | sup={x.support:.3f} conf={x.confidence:.2f} lift={x.lift:.2f}")


# 7. Phân lớp dựa trên luật: CBA, CMAR, CPAR
def covers(X, ante):
    return X[list(ante)].all(axis=1).values

def evaluate(y_true, y_pred):
    kw = dict(pos_label="Approved", zero_division=0)
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred, labels=["Approved", "Rejected"]).ravel()
    return {"F0.5": 100 * fbeta_score(y_true, y_pred, beta=0.5, **kw),
            "Precision": 100 * precision_score(y_true, y_pred, **kw),
            "Recall": 100 * recall_score(y_true, y_pred, **kw),
            "F1": 100 * f1_score(y_true, y_pred, **kw),
            "Accuracy": 100 * accuracy_score(y_true, y_pred), "TP": tp, "FN": fn, "FP": fp, "TN": tn}

def rank(r):
    return r.sort_values(["confidence", "support", "n_items"], ascending=[False, False, True]).reset_index(drop=True)


# CBA
def cba_fit(rules, X, y):
    alive = np.ones(len(y), bool)
    chosen, rule_err, best = [], 0, (np.inf, 0, None)
    for _, r in rank(rules).iterrows():
        m = covers(X, r.ante) & alive
        if not (m & (y == r.cls)).any():
            continue
        chosen.append(r)
        rule_err += (m & (y != r.cls)).sum()
        alive &= ~m
        default = pd.Series(y[alive]).mode()[0] if alive.any() else r.cls
        total_err = rule_err + (alive & (y != default)).sum()
        if total_err < best[0]:
            best = (total_err, len(chosen), default)
        if not alive.any():
            break
    return pd.DataFrame(chosen[:best[1]]), best[2]

def cba_predict(model, X):
    rule_list, default = model
    pred = np.full(len(X), default, dtype=object)
    done = np.zeros(len(X), bool)
    for _, r in rule_list.iterrows():
        m = covers(X, r.ante) & ~done
        pred[m], done = r.cls, done | m
    return pred


# CMAR
def chi2_and_max(X, y, ante, cls):
    a, c, N = covers(X, ante), (y == cls), len(y)
    sa, sc, both = a.sum(), c.sum(), (a & c).sum()
    chi2 = chi2_contingency([[both, sa - both], [sc - both, N - sa - sc + both]], correction=False)[0]
    e = 1/(sa*sc) + 1/(sa*(N-sc)) + 1/((N-sa)*sc) + 1/((N-sa)*(N-sc))
    return chi2, (min(sa, sc) - sa*sc/N) ** 2 * N * e

def cmar_fit(rules, X, y, delta=4):
    r = rank(rules)
    kept = []
    for i, x in r.iterrows():
        if not any(r.at[k, "ante"] < x.ante for k in kept):
            kept.append(i)
    r = r.loc[kept].copy()
    stats = [chi2_and_max(X, y, x.ante, x.cls) for _, x in r.iterrows()]
    r["chi2"] = [s[0] for s in stats]
    r["wchi2"] = [s[0] ** 2 / s[1] for s in stats]
    r = r[(r.chi2 > 3.841) & (r.lift > 1)]
    count, final = np.zeros(len(y)), []
    for i, x in r.iterrows():
        m = covers(X, x.ante) & (count < delta)
        if (m & (y == x.cls)).any():
            final.append(i); count[m] += 1
    return r.loc[final], pd.Series(y).mode()[0]

def cmar_predict(model, X):
    rule_list, default = model
    score = {"Approved": np.zeros(len(X)), "Rejected": np.zeros(len(X))}
    for _, x in rule_list.iterrows():
        score[x.cls] += covers(X, x.ante) * x.wchi2
    pred = np.where(score["Approved"] > score["Rejected"], "Approved", "Rejected").astype(object)
    pred[(score["Approved"] == 0) & (score["Rejected"] == 0)] = default
    return pred


# CPAR
def rule_stats(X, y, ante, cls):
    m = covers(X, ante); n_ante, n_ok = m.sum(), (m & (y == cls)).sum()
    conf = n_ok / n_ante
    return dict(ante=frozenset(ante), cls=cls, n_items=len(ante), support=n_ok / len(y), confidence=conf,
                lift=conf / prior[cls], cf=(conf - prior[cls]) / (1 - prior[cls]),
                laplace=(n_ok + 1) / (n_ante + 2))

def cpar_fit(X, y, min_gain=0.7, decay=2/3, stop=0.05):
    A = X.values.astype(float)
    attr = np.array([c.split("=")[0] for c in X.columns])
    found = set()
    for cls in ["Approved", "Rejected"]:
        pos, neg = (y == cls), (y != cls).astype(float)
        w = pos.astype(float)
        for _ in range(300):
            if w.sum() <= stop * pos.sum():
                break
            ante, mask = [], np.ones(len(y), bool)
            while len(ante) < MAX_ITEMS:
                P, N = (w * mask).sum(), (neg * mask).sum()
                Pj, Nj = (w * mask) @ A, (neg * mask) @ A
                with np.errstate(divide="ignore", invalid="ignore"):
                    gain = Pj * (np.log2(Pj / (Pj + Nj)) - np.log2(P / (P + N)))
                gain = np.nan_to_num(gain, nan=-np.inf)
                gain[np.isin(attr, [a.split("=")[0] for a in ante])] = -np.inf
                j = gain.argmax()
                if gain[j] < min_gain:
                    break
                ante.append(X.columns[j]); mask &= A[:, j].astype(bool)
            if not ante:
                break
            found.add((frozenset(ante), cls))
            w[mask & pos] *= decay
    R = pd.DataFrame([rule_stats(X, y, a, c) for a, c in found])
    R = R[(R.support >= MIN_SUPPORT) & (R.cf >= MIN_CF)]
    return R.sort_values("laplace", ascending=False), pd.Series(y).mode()[0]

def cpar_predict(model, X, k=5):
    rule_list, default = model
    avg = {}
    for cls in ["Approved", "Rejected"]:
        R = rule_list[rule_list.cls == cls]
        if len(R) == 0:
            avg[cls] = np.zeros(len(X)); continue
        M = np.column_stack([covers(X, a) for a in R.ante])
        top = M & (np.cumsum(M, axis=1) <= k)
        n = top.sum(axis=1)
        avg[cls] = np.where(n > 0, (top * R.laplace.values).sum(axis=1) / np.maximum(n, 1), 0)
    pred = np.where(avg["Approved"] > avg["Rejected"], "Approved", "Rejected").astype(object)
    pred[(avg["Approved"] == 0) & (avg["Rejected"] == 0)] = default
    return pred


# 7a. So sánh trên tập validation (F0.5) & độ nhạy chi phí
models, scores = {}, {}
for name, fit, predict in [("CBA", lambda: cba_fit(rules, X_tr, y_tr), cba_predict),
                           ("CMAR", lambda: cmar_fit(rules, X_tr, y_tr), cmar_predict),
                           ("CPAR", lambda: cpar_fit(X_tr, y_tr), cpar_predict)]:
    t = time.time()
    models[name] = fit()
    scores[name] = evaluate(y_val, predict(models[name], X_val))
    print(f"{name}: {len(models[name][0])} luật cuối ({time.time() - t:.0f}s)")
scores = pd.DataFrame(scores).T
print(scores.round(2))

winner = scores.loc[["CBA", "CMAR"], "F0.5"].idxmax()
print(f"\nBước 1 — trong chương trình: {winner} thắng (F0.5 = {scores.loc[winner, 'F0.5']:.2f})")
print(f"Bước 2 — {winner} vs CPAR: {'CPAR' if scores.loc['CPAR', 'F0.5'] > scores.loc[winner, 'F0.5'] else winner} thắng")

cost = pd.DataFrame({f"k={k}": k * scores.FP + scores.FN for k in [0.5, 0.75, 1, 1.5, 2, 3, 5]}).T
cost["Thắng"] = cost.idxmin(axis=1)
print("\nChi phí theo tỉ lệ k = FP/FN (validation):"); print(cost)
cost.drop(columns="Thắng").plot(marker="o", title="Độ nhạy chi phí"); plt.ylabel("k·FP + FN"); plt.show()


# 7b. Kiểm chứng không gian luật chung
pool = set(zip(rules.ante, rules.cls))
for m in ["CBA", "CMAR", "CPAR"]:
    final = set(zip(models[m][0].ante, models[m][0].cls))
    print(f"{m:<5}: {len(final & pool):>4}/{len(final):<4} luật cuối nằm trong tập {len(pool)} luật Apriori")


# 8. Chọn 3 luật → 3 thuộc tính mới
NATIVE = {"CBA": "confidence", "CMAR": "wchi2", "CPAR": "laplace"}

def pick_rules(method, n=3):
    final = models[method][0].sort_values(NATIVE[method], ascending=False)
    pool = final if method == "CPAR" else pd.concat([final, rank(rules)])
    cand = pool[(pool.cls == "Approved") & (pool.n_items >= 2)].drop_duplicates("ante")
    chosen = []
    print(f"\n{method}:")
    for _, r in cand.iterrows():
        where = " AND ".join(f"{i.split('=', 1)[0]} = '{i.split('=', 1)[1]}'" for i in r.ante)
        viol = sql(f"SELECT AVG(violation) AS v FROM v_policy WHERE {where}").v[0]
        m = covers(X_tr, r.ante)
        jac = max([(m & covers(X_tr, c.ante)).sum() / (m | covers(X_tr, c.ante)).sum() for c in chosen], default=0)
        verdict = ("loại: vi phạm chính sách" if viol > MAX_VIOLATION
                   else "loại: chồng lấn" if jac >= 0.5 else "CHỌN")
        print(f"  [{verdict:<24}] vi phạm {viol:5.1%} | {' ∧ '.join(sorted(r.ante))}")
        if verdict == "CHỌN":
            chosen.append(r)
        if len(chosen) == n:
            break
    return chosen

FEATURE_METHODS = [winner, "CPAR"]
picked = {m: pick_rules(m) for m in FEATURE_METHODS}


# 8a. Độ phủ của 3 luật
X_all, y_all = to_items(df).reindex(columns=ITEMS, fill_value=False), df.loan_status.values
for method, chosen in picked.items():
    rows, union = [], np.zeros(len(df), bool)
    for i, r in enumerate(chosen, 1):
        m = covers(X_all, r.ante); union |= m
        rows.append({"feature": f"{method}_rule{i}", "luật": " ∧ ".join(sorted(r.ante)) + " → Approved",
                     "độ phủ %": 100 * m.mean(), "support %": 100 * (m & (y_all == "Approved")).mean(),
                     "confidence": (y_all[m] == "Approved").mean(),
                     "phủ lớp Approved %": 100 * (m & (y_all == "Approved")).sum() / (y_all == "Approved").sum()})
    if not union.any():
        print(f"{method}: không có luật nào qua bộ lọc"); continue
    rows.append({"feature": "Hợp 3 luật", "luật": "", "độ phủ %": 100 * union.mean(),
                 "support %": 100 * (union & (y_all == "Approved")).mean(),
                 "confidence": (y_all[union] == "Approved").mean(),
                 "phủ lớp Approved %": 100 * (union & (y_all == "Approved")).sum() / (y_all == "Approved").sum()})
    print(f"\n=== {method} ==="); display(pd.DataFrame(rows).round(2))
common = {r.ante for r in picked[FEATURE_METHODS[0]]} & {r.ante for r in picked["CPAR"]}
print("Số luật trùng giữa 2 phương pháp:", len(common), "/ 3")


# 9. Phân lớp trên 3 bộ dữ liệu (TH1, TH2, TH3)
def build_X(X_items, chosen):
    X = X_items.astype(int)
    for i, r in enumerate(chosen, 1):
        X[f"rule_{i}"] = covers(X_items, r.ante).astype(int)
    return X

DATASETS = {"TH1 gốc": [], f"TH2 +{FEATURE_METHODS[0]}": picked[FEATURE_METHODS[0]], "TH3 +CPAR": picked["CPAR"]}
ML = {"Logistic Regression": lambda: LogisticRegression(max_iter=2000),
      "Random Forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1),
      "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=42)}

rows = []
for ds, chosen in DATASETS.items():
    Xtr, Xte = build_X(X_tr, chosen), build_X(X_te, chosen)
    for mname, make in ML.items():
        model = make().fit(Xtr, y_tr)
        rows.append({"Bộ dữ liệu": ds, "Mô hình": mname, **evaluate(y_te, model.predict(Xte))})
res = pd.DataFrame(rows)
display(res.round(2))
base = res[res["Bộ dữ liệu"] == "TH1 gốc"].set_index("Mô hình")
print("Chênh lệch so với TH1 (điểm %):")
for ds in list(DATASETS)[1:]:
    cur = res[res["Bộ dữ liệu"] == ds].set_index("Mô hình")
    diff = (cur[["F0.5", "Precision", "Recall", "FP"]] - base[["F0.5", "Precision", "Recall", "FP"]]).round(2)
    print(f"\n{ds}:"); print(diff)

res.pivot(index="Mô hình", columns="Bộ dữ liệu", values="F0.5").plot(kind="bar", rot=0, title="F0.5 lớp Approved trên test")
plt.ylim(res["F0.5"].min() - 3, res["F0.5"].max() + 2); plt.show()


# 10. Xuất hình & bảng cho báo cáo
import os, glob, shutil, subprocess, textwrap, datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = "bao_cao_hinh_bang"
DECIMAL_COMMA = True
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)

def setup_times():
    have = lambda name: any(f.name == name for f in fm.fontManager.ttflist)
    if not have("Times New Roman"):
        cmd = ("apt-get update -qq && echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula "
               "select true | debconf-set-selections && apt-get install -y -qq ttf-mscorefonts-installer fonts-liberation")
        try:
            subprocess.run(cmd, shell=True, capture_output=True, timeout=300)
        except Exception:
            pass
        for p in (glob.glob("/usr/share/fonts/truetype/msttcorefonts/[Tt]imes*.ttf")
                  + glob.glob("/usr/share/fonts/truetype/liberation/LiberationSerif*.ttf")):
            fm.fontManager.addfont(p)
    for name in ["Times New Roman", "Liberation Serif"]:
        if have(name):
            return name
    return "DejaVu Serif"

FONT = setup_times()
print("Font sử dụng:", FONT)
plt.rcParams.update({"pdf.fonttype": 42, "figure.max_open_warning": 0, "font.family": FONT, "mathtext.fontset": "stix", "font.size": 11,
                     "axes.titlesize": 12, "axes.unicode_minus": False})

ITEMS_OUT = []
def add_table(caption, data, note=None):
    ITEMS_OUT.append(("Bảng", caption, data.copy(), note))
def add_figure(caption, fig):
    ITEMS_OUT.append(("Hình", caption, fig, None))

def vn(s):
    return s.replace(",", "X").replace(".", ",").replace("X", ".") if DECIMAL_COMMA else s

def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "Có" if v else "Không"
    if isinstance(v, (int, np.integer)):
        s = f"{v:,}"
    elif isinstance(v, (float, np.floating)):
        s = f"{v:,.0f}" if abs(v) >= 1e5 else f"{v:,.2f}"
    else:
        return str(v)
    return vn(s)

def heatmap(ax, M, xl, yl, f="{:.2f}", cmap="Blues", vmin=None, vmax=None):
    M = np.asarray(M, dtype=float)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl, rotation=45, ha="right")
    ax.set_yticks(range(len(yl))); ax.set_yticklabels(yl)
    lo, hi = (np.nanmin(M) if vmin is None else vmin), (np.nanmax(M) if vmax is None else vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            t = vn(f.format(M[i, j]))
            dark = (M[i, j] - lo) / ((hi - lo) or 1) > 0.6
            ax.text(j, i, t, ha="center", va="center", fontsize=8 if M.size > 9 else 11, color="white" if dark else "black")
    return im

def bar_labels(ax, fmt_str="{:.1f}"):
    for p in ax.patches:
        ax.annotate(vn(fmt_str.format(p.get_height())), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")

def legend_below(ax, ncol=3, headroom=1.15):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=ncol, frameon=False)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, lo + (hi - lo) * headroom)

def flowchart():
    B = [(190, 40, 300, 44, "Dữ liệu gốc (45.000 hồ sơ)", None), (190, 110, 300, 44, "EDA & làm sạch dữ liệu", None),
         (190, 180, 300, 56, "Tiền xử lý & kiểm tra đa cộng tuyến", "Rời rạc hoá, Pearson, Cramér's V"),
         (60, 280, 240, 56, "Xây dựng Data Warehouse", "Star schema, view chính sách"),
         (380, 280, 240, 56, "Chia dữ liệu train / val / test", "Tỉ lệ 60 / 20 / 20"),
         (60, 370, 240, 56, "OLAP & truy vấn thực tế", "Slice, ROLLUP, CUBE, biểu đồ"),
         (380, 370, 240, 56, "Khai phá luật kết hợp", "Apriori vs FP-Growth"),
         (380, 460, 240, 56, "Phân lớp dựa trên luật", "CBA, CMAR vs CPAR (F0.5)"),
         (380, 550, 240, 56, "Chọn 3 luật → 3 thuộc tính mới", "Lọc chính sách, tính độ phủ"),
         (380, 640, 240, 56, "Phân lớp trên 3 bộ dữ liệu", "TH1, TH2, TH3 × LR, RF, GB"),
         (190, 740, 300, 44, "Đánh giá & kết luận", None)]
    A = [[(340, 84), (340, 110)], [(340, 154), (340, 180)],
         [(340, 236), (340, 258), (180, 258), (180, 280)], [(340, 258), (500, 258), (500, 280)],
         [(180, 336), (180, 370)], [(500, 336), (500, 370)], [(500, 426), (500, 460)],
         [(500, 516), (500, 550)], [(500, 606), (500, 640)],
         [(180, 426), (180, 762), (190, 762)], [(500, 696), (500, 762), (490, 762)]]
    fig, ax = plt.subplots(figsize=(5.9, 7.75))
    ax.set_xlim(45, 635); ax.set_ylim(800, 25); ax.axis("off")
    for x, y, w, h, t, s in B:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=5",
                                    facecolor="white", edgecolor="black", lw=0.9))
        if s:
            ax.text(x + w / 2, y + h * .36, t, ha="center", va="center", fontsize=9, weight="bold")
            ax.text(x + w / 2, y + h * .70, s, ha="center", va="center", fontsize=8, style="italic")
        else:
            ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9, weight="bold")
    def line(pts, ls="-"):
        for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
            ax.plot([x1, x2], [y1, y2], color="black", lw=0.9, linestyle=ls)
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=9, color="black",
                                     lw=0.9, linestyle=ls, shrinkA=0, shrinkB=0))
    for p in A:
        line(p)
    line([(300, 320), (340, 320), (340, 578), (380, 578)], ls=(0, (4, 3)))
    ax.text(334, 562, "kiểm tra chính sách", ha="right", va="center", fontsize=8, style="italic")
    return fig

def star_schema():
    HDR, ROW, MARK = 26, 20, 30
    fig, ax = plt.subplots(figsize=(6.65, 4.65))
    ax.set_xlim(25, 690); ax.set_ylim(485, 20); ax.axis("off")
    def table(x, y, w, name, rows):
        h = HDR + ROW * len(rows)
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="black", lw=0.9))
        ax.add_patch(Rectangle((x, y), w, HDR, facecolor="white", edgecolor="black", lw=0.9))
        ax.text(x + w / 2, y + HDR / 2, name, ha="center", va="center", fontsize=9, weight="bold")
        ax.plot([x + MARK] * 2, [y + HDR, y + h], color="black", lw=0.6)
        for i, (m, c, strong) in enumerate(rows):
            ry = y + HDR + ROW * i
            if i:
                ax.plot([x, x + w], [ry, ry], color="black", lw=0.9 if strong else 0.3)
            ax.text(x + MARK / 2, ry + ROW / 2, m, ha="center", va="center", fontsize=7, weight="bold")
            ax.text(x + MARK + 6, ry + ROW / 2, c, ha="left", va="center", fontsize=8)
        return h
    FX, FY, FW = 245, 45, 225
    rows = ([("FK", c, False) for c in ["education_id", "home_id", "intent_id", "gender_id", "prev_default_id", "status_id"]]
            + [("M", c, i == 0) for i, c in enumerate(["person_income", "loan_amnt", "loan_int_rate",
                                                         "loan_percent_income", "credit_score", "person_age"])]
            + [("DD", c + "_band", i == 0) for i, c in enumerate(BINS)])
    fh = table(FX, FY, FW, "fact_loan", rows)
    DW_, DH = 150, HDR + 2 * ROW
    for side, names, x in [("L", ["dim_education", "dim_home", "dim_intent"], 40),
                           ("R", ["dim_gender", "dim_prev_default", "dim_status"], 525)]:
        for name, cy in zip(names, [FY + fh * .18, FY + fh * .5, FY + fh * .82]):
            table(x, cy - DH / 2, DW_, name, [("PK", "id", False), ("", "value", False)])
            x1, x2 = (x + DW_, FX) if side == "L" else (FX + FW, x)
            l1, l2 = ("1", "n") if side == "L" else ("n", "1")
            ax.text(x1 + 5, cy - 5, l1, fontsize=8, ha="left", va="bottom")
            ax.text(x2 - 5, cy - 5, l2, fontsize=8, ha="right", va="bottom")
            ax.plot([x1, x2], [cy, cy], color="black", lw=0.9)
    ax.text(357, FY + fh + 22, "PK: khoá chính   FK: khoá ngoại   M: độ đo (measure)   DD: chiều suy biến (degenerate dimension)",
            ha="center", va="center", fontsize=7.5, style="italic")
    return fig

add_figure("Quy trình nghiên cứu của đề tài", flowchart())
add_figure("Mô hình hình sao (star schema) của kho dữ liệu hồ sơ vay", star_schema())

raw = pd.read_csv("loan_data.csv")
n_bad = int(((raw.person_age > 80) | (raw.person_emp_exp > 60)).sum())
n_dup = len(raw[(raw.person_age <= 80) & (raw.person_emp_exp <= 60)]) - len(df)
add_table("Tổng quan bộ dữ liệu và kết quả làm sạch", pd.DataFrame({
    "Chỉ tiêu": ["Số dòng ban đầu", "Số cột", "Số giá trị thiếu", "Dòng phi lý bị loại (tuổi > 80 hoặc kinh nghiệm > 60)",
                 "Dòng trùng lặp bị loại", "Số dòng sau làm sạch"],
    "Giá trị": [raw.shape[0], raw.shape[1], int(raw.isna().sum().sum()), n_bad, n_dup, len(df)]}))

desc = df[num_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
desc.columns = ["Trung bình", "Độ lệch chuẩn", "Min", "Q1", "Trung vị", "Q3", "Max"]
add_table("Thống kê mô tả các biến số (sau làm sạch)", desc.rename_axis("Biến").reset_index())

cat_cols = ["person_gender", "person_education", "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]
add_table("Mô tả các biến phân loại", pd.DataFrame([{
    "Biến": c, "Số giá trị": df[c].nunique(), "Giá trị phổ biến nhất": df[c].value_counts().index[0],
    "Tỉ lệ (%)": 100 * df[c].value_counts(normalize=True).iloc[0]} for c in cat_cols]))

cls = df.loan_status.value_counts()
add_table("Phân phối biến mục tiêu loan_status", pd.DataFrame({"Lớp": cls.index, "Số hồ sơ": cls.values,
                                                                "Tỉ lệ (%)": 100 * cls.values / cls.sum()}))
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(cls.index, cls.values, color=["#C1502E", "#4C8C6B"] if cls.index[0] == "Rejected" else ["#4C8C6B", "#C1502E"])
bar_labels(ax, "{:,.0f}"); ax.set_ylabel("Số hồ sơ"); ax.set_title("Phân phối lớp loan_status")
add_figure("Phân phối biến mục tiêu loan_status", fig)

for col, title in [("credit_score", "điểm tín dụng"), ("loan_percent_income", "tỉ lệ khoản vay / thu nhập"),
                   ("person_income", "thu nhập")]:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    data_col = df[col] if col != "person_income" else df[col].clip(upper=df[col].quantile(0.99))
    for s, c in [("Approved", "#4C8C6B"), ("Rejected", "#C1502E")]:
        ax.hist(data_col[df.loan_status == s], bins=30, alpha=0.6, label=s, color=c)
    ax.set_xlabel(col + (" (cắt ở phân vị 99%)" if col == "person_income" else "")); ax.set_ylabel("Số hồ sơ")
    ax.legend(); ax.set_title(f"Phân phối {title} theo loan_status")
    add_figure(f"Phân phối {title} theo loan_status", fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col, t in zip(axes, ["previous_loan_defaults_on_file", "person_home_ownership", "loan_intent"],
                      ["Lịch sử vỡ nợ", "Hình thức sở hữu nhà", "Mục đích vay"]):
    r = df.groupby(col).loan_status.apply(lambda s: 100 * (s == "Approved").mean()).sort_values()
    ax.bar(r.index, r.values, color="#1F3D3A"); bar_labels(ax)
    ax.set_title(t); ax.set_ylabel("% được duyệt"); ax.tick_params(axis="x", rotation=35)
add_figure("Tỉ lệ duyệt theo lịch sử vỡ nợ, sở hữu nhà và mục đích vay", fig)

fig, ax = plt.subplots(figsize=(7.5, 6))
im = heatmap(ax, corr.values, num_cols, num_cols, cmap="RdBu_r", vmin=-1, vmax=1); plt.colorbar(im)
ax.set_title("Ma trận tương quan Pearson")
add_figure("Ma trận tương quan Pearson giữa các biến số", fig)
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(num_cols) for b in num_cols[i + 1:]]
pairs = pd.DataFrame(sorted(pairs, key=lambda t: -abs(t[2]))[:6], columns=["Biến 1", "Biến 2", "Hệ số r"])
pairs["Đánh giá"] = np.where(pairs["Hệ số r"].abs() >= 0.8, "Đa cộng tuyến cao -> giữ 1 biến", "Chấp nhận được")
add_table("Các cặp biến số có tương quan mạnh nhất", pairs)

add_table("Ngưỡng rời rạc hoá các biến số (ngưỡng nghiệp vụ)", pd.DataFrame([{
    "Biến": c, "Mốc cắt": " | ".join(fmt(e) for e in edges[1:-1]), "Nhãn": ", ".join(labels),
    "Phân bố (%)": ", ".join(fmt(100 * (df[c + "_band"] == l).mean()) for l in labels)}
    for c, (edges, labels) in BINS.items()]))

fig, ax = plt.subplots(figsize=(8.5, 7))
im = heatmap(ax, V.values, ATTRS, ATTRS, cmap="Oranges", vmin=0, vmax=1); plt.colorbar(im)
ax.set_title("Cramér's V giữa các thuộc tính rời rạc")
add_figure("Ma trận Cramér's V giữa các thuộc tính rời rạc", fig)
vp = [(a, b, V.loc[a, b]) for i, a in enumerate(ATTRS) for b in ATTRS[i + 1:]]
vp = pd.DataFrame(sorted(vp, key=lambda t: -t[2])[:6], columns=["Thuộc tính 1", "Thuộc tính 2", "Cramér's V"])
add_table("Các cặp thuộc tính rời rạc có Cramér's V cao nhất", vp, note="Ngưỡng cảnh báo: V ≥ 0,5")

add_table("Kết quả chia dữ liệu", pd.DataFrame([{"Tập": n, "Số hồ sơ": len(d), "Tỉ trọng (%)": 100 * len(d) / len(df),
    "% Approved": 100 * (d.loan_status == "Approved").mean()} for n, d in [("Train", train), ("Validation", val), ("Test", test)]]))

tbls = sql("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY type, name")
tbls["Số dòng"] = [sql(f"SELECT COUNT(*) AS n FROM {t}").n[0] for t in tbls.name]
tbls["Số cột"] = [len(sql(f"SELECT * FROM {t} LIMIT 1").columns) for t in tbls.name]
tbls["type"] = tbls["type"].map({"table": "Bảng", "view": "View"})
add_table("Các bảng và view trong kho dữ liệu", tbls.rename(columns={"name": "Tên", "type": "Loại"}))

q = sql(f"SELECT previous_loan_defaults_on_file AS prev_default, COUNT(*) AS n, {PCT} AS pct_approved FROM v_loan GROUP BY prev_default")
add_table("OLAP Slice – tỉ lệ duyệt theo lịch sử vỡ nợ", q)
fig, ax = plt.subplots(figsize=(5, 3.6)); ax.bar(q.prev_default, q.pct_approved, color="#1F3D3A"); bar_labels(ax)
ax.set_ylabel("% được duyệt"); ax.set_xlabel("previous_loan_defaults_on_file"); ax.set_title("Slice: % duyệt theo lịch sử vỡ nợ")
add_figure("OLAP Slice – tỉ lệ duyệt theo lịch sử vỡ nợ", fig)

q = sql('''SELECT person_education AS edu, person_home_ownership AS home, COUNT(*) AS n, ROUND(AVG(loan_amnt)) AS avg_loan
  FROM v_loan GROUP BY edu, home
  UNION ALL SELECT person_education, 'ALL', COUNT(*), ROUND(AVG(loan_amnt)) FROM v_loan GROUP BY person_education
  UNION ALL SELECT 'ALL', 'ALL', COUNT(*), ROUND(AVG(loan_amnt)) FROM v_loan''')
add_table("OLAP ROLLUP (học vấn → sở hữu nhà) – khoản vay bình quân", q)
fig, ax = plt.subplots(figsize=(8, 4.2))
q[(q.edu != "ALL") & (q.home != "ALL")].pivot(index="edu", columns="home", values="avg_loan").plot(kind="bar", rot=0, ax=ax)
ax.set_ylabel("Khoản vay bình quân"); ax.set_xlabel("Học vấn"); ax.set_title("ROLLUP: khoản vay bình quân theo học vấn × sở hữu nhà")
legend_below(ax, ncol=4)
add_figure("OLAP ROLLUP – khoản vay bình quân theo học vấn và sở hữu nhà", fig)

q = sql(f'''SELECT loan_intent AS intent, person_gender AS gender, {PCT} AS pct FROM v_loan GROUP BY intent, gender
  UNION ALL SELECT loan_intent, 'ALL', {PCT} FROM v_loan GROUP BY loan_intent
  UNION ALL SELECT 'ALL', person_gender, {PCT} FROM v_loan GROUP BY person_gender
  UNION ALL SELECT 'ALL', 'ALL', {PCT} FROM v_loan''')
cube_t = q.pivot(index="intent", columns="gender", values="pct")
add_table("OLAP CUBE (mục đích vay × giới tính) – % được duyệt", cube_t.rename_axis(None, axis=1).rename_axis("Mục đích vay").reset_index())
fig, ax = plt.subplots(figsize=(5.5, 5)); im = heatmap(ax, cube_t.values, list(cube_t.columns), list(cube_t.index), f="{:.1f}", cmap="RdYlGn")
plt.colorbar(im, label="% duyệt"); ax.set_title("CUBE: % duyệt theo mục đích vay × giới tính")
add_figure("OLAP CUBE – % được duyệt theo mục đích vay và giới tính", fig)

q = sql("SELECT loan_intent, COUNT(*) AS so_ho_so, SUM(loan_amnt) AS du_no FROM v_loan WHERE loan_status='Approved' GROUP BY loan_intent ORDER BY du_no DESC")
q["ty_trong_pct"] = 100 * q.du_no / q.du_no.sum()
add_table("Truy vấn 1 – Dư nợ đã duyệt theo mục đích vay", q)
fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(q.loan_intent, q.du_no / 1e6, color="#1F3D3A"); bar_labels(ax)
ax.set_ylabel("Triệu USD"); ax.tick_params(axis="x", rotation=30); ax.set_title("Dư nợ đã duyệt theo mục đích vay")
add_figure("Truy vấn 1 – Dư nợ đã duyệt theo mục đích vay", fig)

ORDER = ["Poor", "Fair", "Good", "VeryGood"]
q = sql(f"SELECT credit_score_band, previous_loan_defaults_on_file AS prev_default, COUNT(*) AS n, {PCT} AS pct FROM v_loan GROUP BY credit_score_band, prev_default")
m = q.pivot(index="credit_score_band", columns="prev_default", values="pct").loc[ORDER]
add_table("Truy vấn 2 – % được duyệt theo điểm tín dụng × lịch sử vỡ nợ", m.rename_axis(None, axis=1).rename_axis("Điểm tín dụng").reset_index())
fig, ax = plt.subplots(figsize=(6.5, 4)); m.plot(kind="bar", rot=0, ax=ax); bar_labels(ax)
ax.set_ylabel("% được duyệt"); ax.set_xlabel("Nhóm điểm tín dụng"); ax.set_title("% duyệt theo điểm tín dụng × lịch sử vỡ nợ")
legend_below(ax, ncol=2)
add_figure("Truy vấn 2 – % được duyệt theo điểm tín dụng và lịch sử vỡ nợ", fig)

q = sql(f"SELECT loan_intent || ' / ' || person_home_ownership AS nhom, COUNT(*) AS n, {PCT} AS pct FROM v_loan GROUP BY loan_intent, person_home_ownership HAVING n >= 500 ORDER BY pct LIMIT 5")
add_table("Truy vấn 3 – 5 nhóm (mục đích, sở hữu nhà) có tỉ lệ duyệt thấp nhất (n ≥ 500)", q)
fig, ax = plt.subplots(figsize=(7, 3.6)); ax.barh(q.nhom, q.pct, color="#C1502E"); ax.invert_yaxis()
ax.set_xlabel("% được duyệt"); ax.set_title("5 nhóm có tỉ lệ duyệt thấp nhất")
add_figure("Truy vấn 3 – 5 nhóm có tỉ lệ duyệt thấp nhất", fig)

q = sql(f"SELECT credit_score_band, violation, COUNT(*) AS n, {PCT} AS pct FROM v_policy WHERE previous_loan_defaults_on_file='No' GROUP BY credit_score_band, violation")
m = q.pivot(index="credit_score_band", columns="violation", values="pct").loc[ORDER]; m.columns = ["Không vi phạm", "Vi phạm"]
add_table("Truy vấn 4 – % được duyệt của hồ sơ vi phạm / không vi phạm năng lực trả nợ", m.rename_axis("Điểm tín dụng").reset_index(),
          note=f"Nhóm chưa từng vỡ nợ. Vi phạm: thu nhập < {POLICY_MIN_INCOME:,} hoặc khoản vay > {POLICY_MAX_LPI:.0%} thu nhập")
fig, ax = plt.subplots(figsize=(6.5, 4)); m.plot(kind="bar", rot=0, ax=ax, color=["#4C8C6B", "#C1502E"]); bar_labels(ax)
ax.set_ylabel("% được duyệt"); ax.set_xlabel("Nhóm điểm tín dụng"); ax.set_title("% duyệt: vi phạm vs không vi phạm chính sách")
legend_below(ax, ncol=2)
add_figure("Truy vấn 4 – % được duyệt của hồ sơ vi phạm và không vi phạm năng lực trả nợ", fig)

add_table("So sánh Apriori và FP-Growth", pd.DataFrame([{"Thuật toán": k, "Số itemset": len(v[0]), "Thời gian (giây)": v[1]}
          for k, v in results.items()]), note=f"Hai thuật toán cho cùng tập itemset: {'Có' if same else 'Không'}")
rc = rules.cls.value_counts()
add_table("Ràng buộc khai phá và số luật lớp thu được", pd.DataFrame({
    "Chỉ tiêu": ["min support", "min certainty factor", "Số điều kiện tối đa ở vế trái", "Luật → Approved", "Luật → Rejected", "Tổng số luật"],
    "Giá trị": [MIN_SUPPORT, MIN_CF, MAX_ITEMS, int(rc.get("Approved", 0)), int(rc.get("Rejected", 0)), len(rules)]}))
top = pd.concat([rules[rules.cls == c].nlargest(5, "lift") for c in ["Approved", "Rejected"]])
add_table("Ví dụ minh hoạ – 5 luật có lift cao nhất mỗi lớp", pd.DataFrame({
    "Luật": [" AND ".join(sorted(a)) + " → " + c for a, c in zip(top.ante, top.cls)],
    "Support": top.support.values, "Confidence": top.confidence.values, "Lift": top.lift.values, "CF": top.cf.values}))
fig, ax = plt.subplots(figsize=(7, 4.8))
for c, mk in [("Approved", "o"), ("Rejected", "x")]:
    s = rules[rules.cls == c]
    sc = ax.scatter(s.support, s.confidence, c=s.lift, cmap="viridis", marker=mk, s=18, label=c,
                    vmin=rules.lift.min(), vmax=rules.lift.max())
plt.colorbar(sc, label="Lift"); ax.legend(title="Vế phải"); ax.set_xlabel("Support"); ax.set_ylabel("Confidence")
ax.set_title("Support – confidence – lift của các luật lớp")
add_figure("Phân bố support, confidence và lift của các luật lớp", fig)

sc_t = scores.copy(); sc_t.insert(0, "Số luật cuối", [len(models[m][0]) for m in sc_t.index])
for c in ["TP", "FN", "FP", "TN"]:
    sc_t[c] = sc_t[c].astype(int)
add_table("So sánh CBA, CMAR và CPAR trên tập validation", sc_t.rename_axis("Thuật toán").reset_index(),
          note=f"Bước 1 (trong chương trình): {winner} thắng; bước 2: {winner} vs CPAR – chọn theo F0.5")
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
for ax, mth in zip(axes, scores.index):
    cm = np.array([[scores.loc[mth, "TP"], scores.loc[mth, "FN"]], [scores.loc[mth, "FP"], scores.loc[mth, "TN"]]])
    heatmap(ax, cm, ["Dự đoán A", "Dự đoán R"], ["Thực tế A", "Thực tế R"], f="{:,.0f}")
    ax.set_title(mth); ax.tick_params(axis="x", rotation=0); ax.set_xticklabels(["Dự đoán A", "Dự đoán R"], ha="center")
fig.tight_layout()
add_figure("Ma trận nhầm lẫn của CBA, CMAR và CPAR trên tập validation (A: Approved, R: Rejected)", fig)
fig, ax = plt.subplots(figsize=(7, 4))
scores[["F0.5", "Precision", "Recall"]].plot(kind="bar", rot=0, ax=ax); bar_labels(ax)
ax.set_ylim(0, 100); ax.set_ylabel("%"); ax.set_title("F0.5, precision, recall lớp Approved (validation)")
legend_below(ax, ncol=3)
add_figure("So sánh F0.5, precision và recall của CBA, CMAR, CPAR", fig)
add_table("Phân tích độ nhạy chi phí (chi phí = k·FP + FN, tập validation)", cost.rename_axis("Tỉ lệ k").reset_index())
fig, ax = plt.subplots(figsize=(6.5, 4))
cost.drop(columns="Thắng").plot(marker="o", ax=ax); ax.set_xlabel("k = chi phí 1 FP / chi phí 1 FN"); ax.set_ylabel("k·FP + FN")
ax.set_title("Độ nhạy chi phí theo tỉ lệ k")
add_figure("Độ nhạy chi phí của CBA, CMAR và CPAR theo tỉ lệ k", fig)
add_table("Kiểm chứng ba thuật toán dùng chung không gian luật Apriori", pd.DataFrame([{
    "Thuật toán": m, "Số luật cuối": len(models[m][0]),
    "Thuộc tập luật Apriori": len(set(zip(models[m][0].ante, models[m][0].cls)) & pool),
    "Tổng số luật Apriori": len(pool)} for m in ["CBA", "CMAR", "CPAR"]]))

def viol(ante):
    where = " AND ".join(f"{i.split('=', 1)[0]} = '{i.split('=', 1)[1]}'" for i in ante)
    return 100 * sql(f"SELECT AVG(violation) AS v FROM v_policy WHERE {where}").v[0]
add_table("Các luật được chọn làm thuộc tính mới", pd.DataFrame([{
    "Phương pháp": m, "Thuộc tính": f"rule_{i}", "Luật": " AND ".join(sorted(r.ante)) + " → Approved",
    "Độ đo xếp hạng": NATIVE[m], "% vi phạm chính sách": viol(r.ante)}
    for m in FEATURE_METHODS for i, r in enumerate(picked[m], 1)]), note=f"Luật bị loại nếu > {MAX_VIOLATION:.0%} hồ sơ phủ vi phạm")
cov_rows = []
for m in FEATURE_METHODS:
    union = np.zeros(len(df), bool)
    for i, r in enumerate(picked[m], 1):
        mk = covers(X_all, r.ante); union |= mk
        cov_rows.append({"Phương pháp": m, "Luật": f"rule_{i}", "Độ phủ (%)": 100 * mk.mean(),
                         "Support (%)": 100 * (mk & (y_all == "Approved")).mean(), "Confidence": (y_all[mk] == "Approved").mean(),
                         "Phủ lớp Approved (%)": 100 * (mk & (y_all == "Approved")).sum() / (y_all == "Approved").sum()})
    if union.any():
        cov_rows.append({"Phương pháp": m, "Luật": "Hợp 3 luật", "Độ phủ (%)": 100 * union.mean(),
                         "Support (%)": 100 * (union & (y_all == "Approved")).mean(), "Confidence": (y_all[union] == "Approved").mean(),
                         "Phủ lớp Approved (%)": 100 * (union & (y_all == "Approved")).sum() / (y_all == "Approved").sum()})
cov_t = pd.DataFrame(cov_rows)
add_table("Độ phủ của 3 luật mỗi phương pháp trên toàn bộ dữ liệu", cov_t)
fig, ax = plt.subplots(figsize=(8, 4))
lab = cov_t["Phương pháp"] + " " + cov_t["Luật"]
xx = np.arange(len(cov_t))
ax.bar(xx - 0.2, cov_t["Độ phủ (%)"], 0.4, label="Độ phủ dữ liệu (%)", color="#8E8E8E")
ax.bar(xx + 0.2, cov_t["Phủ lớp Approved (%)"], 0.4, label="Phủ lớp Approved (%)", color="#1F3D3A"); bar_labels(ax)
ax.set_xticks(xx); ax.set_xticklabels(lab, rotation=30, ha="right"); ax.set_title("Độ phủ của các luật được chọn")
ax.legend(loc="upper left", frameon=False); ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
add_figure("Độ phủ dữ liệu và độ phủ lớp Approved của các luật được chọn", fig)

add_table("Kết quả phân lớp trên tập test (lớp Approved)", res[["Bộ dữ liệu", "Mô hình", "F0.5", "Precision", "Recall", "F1", "Accuracy"]])
add_table("Ma trận nhầm lẫn trên tập test", res[["Bộ dữ liệu", "Mô hình", "TP", "FN", "FP", "TN"]])
base = res[res["Bộ dữ liệu"] == "TH1 gốc"].set_index("Mô hình")
dif = []
for ds in list(DATASETS)[1:]:
    cur = res[res["Bộ dữ liệu"] == ds].set_index("Mô hình")
    for mname in ML:
        dif.append({"Bộ dữ liệu": ds, "Mô hình": mname, **{f"Δ {k}": cur.loc[mname, k] - base.loc[mname, k]
                                                       for k in ["F0.5", "Precision", "Recall"]},
                    "Δ FP": int(cur.loc[mname, "FP"] - base.loc[mname, "FP"])})
add_table("Chênh lệch so với TH1 (điểm %; FP: số hồ sơ)", pd.DataFrame(dif))
for metric in ["F0.5", "Precision"]:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    piv = res.pivot(index="Mô hình", columns="Bộ dữ liệu", values=metric)[list(DATASETS)]
    piv.plot(kind="bar", rot=0, ax=ax, color=["#8E8E8E", "#1F3D3A", "#C1502E"]); bar_labels(ax)
    ax.set_ylim(res[metric].min() - 5, res[metric].max() + 3); ax.set_ylabel("%"); ax.set_title(f"{metric} lớp Approved trên tập test")
    legend_below(ax, ncol=3, headroom=1.05)
    add_figure(f"{metric} lớp Approved trên tập test theo bộ dữ liệu và mô hình", fig)
fig, axes = plt.subplots(3, 3, figsize=(11, 10))
for i, ds in enumerate(DATASETS):
    for j, mname in enumerate(ML):
        r = res[(res["Bộ dữ liệu"] == ds) & (res["Mô hình"] == mname)].iloc[0]
        heatmap(axes[i, j], [[r.TP, r.FN], [r.FP, r.TN]], ["Dự đoán A", "Dự đoán R"], ["Thực tế A", "Thực tế R"], f="{:,.0f}")
        axes[i, j].set_xticklabels(["Dự đoán A", "Dự đoán R"], rotation=0, ha="center")
        axes[i, j].set_title(f"{ds} – {mname}", fontsize=10)
fig.tight_layout()
add_figure("Ma trận nhầm lẫn của 9 tổ hợp bộ dữ liệu × mô hình trên tập test (A: Approved, R: Rejected)", fig)

A4 = (8.27, 11.69)
ROWS_PER_PAGE = 30
def chunks(d):
    return [d.iloc[i:i + ROWS_PER_PAGE] for i in range(0, max(len(d), 1), ROWS_PER_PAGE)]

plan, nb, nh = [], 0, 0
for kind, cap, obj, note in ITEMS_OUT:
    if kind == "Bảng":
        nb += 1; plan.append((kind, nb, cap, obj, note, len(chunks(obj))))
    else:
        nh += 1; plan.append((kind, nh, cap, obj, note, 1))
TOC_LINES = 46
toc_pages = int(np.ceil(len(plan) / TOC_LINES))
page_no, starts = toc_pages + 1, []
for p in plan:
    starts.append(page_no); page_no += p[5]

def footer(fig, n):
    fig.text(0.5, 0.025, str(n), ha="center", fontsize=10)

def table_page(pdf, title, d, note, n):
    d = d.copy()
    cols = [str(c) for c in d.columns]
    cells = [[textwrap.fill(fmt(v), 46) for v in row] for row in d.itertuples(index=False)]
    lines = [[s.count("\n") + 1 for s in row] for row in cells]
    width = [max([len(c)] + [max(len(x) for x in r[j].split("\n")) for r in cells]) for j, c in enumerate(cols)]
    total = sum(width); fs = 10 if total < 80 else 9 if total < 105 else 8
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.955, title, ha="center", va="top", fontsize=12, weight="bold", wrap=True)
    ax = fig.add_axes([0.06, 0.07, 0.88, 0.85]); ax.axis("off")
    t = ax.table(cellText=cells, colLabels=cols, colWidths=[w / total for w in width], loc="upper center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(fs)
    unit = fs * 1.75 / 72 / (0.85 * A4[1])
    for (r, c), cell in t.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_height(unit * (lines[r - 1][c] if r > 0 else 1.3))
        cell.get_text().set_fontfamily(FONT)
        if r == 0:
            cell.get_text().set_weight("bold")
    if note:
        fig.text(0.06, 0.06, "Ghi chú: " + note, fontsize=9, style="italic", va="bottom")
    footer(fig, n); pdf.savefig(fig); plt.close(fig)

pdf_path = f"{OUT}/hinh_va_bang_bao_cao.pdf"
with PdfPages(pdf_path) as pdf:
    for tp in range(toc_pages):
        fig = plt.figure(figsize=A4)
        if tp == 0:
            fig.text(0.5, 0.95, "DANH MỤC HÌNH VÀ BẢNG", ha="center", fontsize=15, weight="bold")
            fig.text(0.5, 0.925, f"Xuất ngày {datetime.date.today():%d/%m/%Y} – font {FONT}", ha="center", fontsize=10, style="italic")
        for k, ((kind, idx, cap, *_), st) in enumerate(list(zip(plan, starts))[tp * TOC_LINES:(tp + 1) * TOC_LINES]):
            y = 0.89 - k * 0.0182
            fig.text(0.08, y, f"{kind} {idx}. {cap}"[:105], fontsize=9.5)
            fig.text(0.92, y, str(st), fontsize=9.5, ha="right")
        footer(fig, tp + 1); pdf.savefig(fig); plt.close(fig)
    for (kind, idx, cap, obj, note, npg), st in zip(plan, starts):
        title = f"{kind} {idx}. {cap}"
        if kind == "Bảng":
            obj.to_csv(f"{OUT}/Bang_{idx:02d}.csv", index=False, encoding="utf-8-sig")
            for c, part in enumerate(chunks(obj)):
                table_page(pdf, title + (" (tiếp)" if c else ""), part, note if c == npg - 1 else None, st + c)
        else:
            png = f"{OUT}/Hinh_{idx:02d}.png"
            obj.savefig(png, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(obj)
            img = plt.imread(png); h, w = img.shape[:2]
            wf = 0.86; hf = wf * A4[0] / A4[1] * h / w
            if hf > 0.80:
                wf, hf = wf * 0.80 / hf, 0.80
            page = plt.figure(figsize=A4)
            ax = page.add_axes([(1 - wf) / 2, 0.93 - hf, wf, hf]); ax.imshow(img); ax.axis("off")
            page.text(0.5, 0.93 - hf - 0.02, title, ha="center", va="top", fontsize=12, weight="bold", wrap=True)
            footer(page, st); pdf.savefig(page); plt.close(page)

shutil.make_archive(OUT, "zip", OUT)
print(f"Đã xuất {nh} hình và {nb} bảng -> {pdf_path} ({page_no - 1} trang) và {OUT}.zip")
try:
    from google.colab import files
    files.download(f"{OUT}.zip")
except ImportError:
    pass


# 11. Demo web
try:
    import gradio as gr
    gr.components.Component
except (ImportError, AttributeError):
    raise RuntimeError("Gradio chưa cài hoặc đang lỗi. Chạy `!pip install -q -U gradio`, "
                       "rồi Runtime -> Restart session -> Run all.")

BEST = scores["F0.5"].idxmax()
BEST_PREDICT = {"CBA": cba_predict, "CMAR": cmar_predict, "CPAR": cpar_predict}[BEST]
SAFE_MIN_INCOME, SAFE_MAX_LPI, SAFE_MIN_SCORE = 100_000, 0.10, 580
FIRST_MIN_EXP, FIRST_MAX_HIST = 2, 4

def to_items_one(age, income, loan, rate, score, gender, edu, home, intent, prev):
    lpi = round(loan / income, 2) if income else 0
    raw = pd.DataFrame([dict(person_age=age, person_income=income, loan_amnt=loan, loan_int_rate=rate,
                             loan_percent_income=lpi, credit_score=score, person_gender=gender,
                             person_education=edu, person_home_ownership=home, loan_intent=intent,
                             previous_loan_defaults_on_file=prev)])
    for col, (edges, labels) in BINS.items():
        raw[col + "_band"] = pd.cut(raw[col], bins=edges, labels=labels, right=False).astype(str)
    return to_items(raw).reindex(columns=ITEMS, fill_value=False), lpi

def matched_rules(method, X):
    R = models[method][0]
    hit = R[[covers(X, a)[0] for a in R.ante]].sort_values(NATIVE[method], ascending=False).head(10)
    return pd.DataFrame({"Luật khớp": [" AND ".join(sorted(a)) + " → " + c for a, c in zip(hit.ante, hit.cls)],
                         "Confidence": hit.confidence.round(2), "Lift": hit.lift.round(2),
                         f"Xếp hạng ({NATIVE[method]})": hit[NATIVE[method]].round(3)})

def demo_predict(age, income, loan, rate, score, gender, edu, home, intent, prev, emp, hist):
    X, lpi = to_items_one(age, income, loan, rate, score, gender, edu, home, intent, prev)
    model_pred = BEST_PREDICT(models[BEST], X)[0]
    if prev == "Yes":
        pred, by = "Rejected", "Luật chuyên gia – từng vỡ nợ"
    elif income < POLICY_MIN_INCOME or lpi > POLICY_MAX_LPI:
        pred, by = "Rejected", "Luật chuyên gia – vi phạm năng lực trả nợ"
    elif income >= SAFE_MIN_INCOME and lpi <= SAFE_MAX_LPI and score >= SAFE_MIN_SCORE:
        pred, by = "Approved", "Luật chuyên gia – hồ sơ an toàn"
    else:
        pred, by = model_pred, f"Mô hình {BEST}"
    text = (f"### {'✅' if pred == 'Approved' else '❌'} Kết quả: **{pred}**  \n"
            f"Quyết định bởi: {by}  \n(Mô hình {BEST} dự đoán: {model_pred})\n")
    text += f"\n**Khoản vay / thu nhập:** {lpi:.0%}  \n**Hồ sơ sau rời rạc hoá:** " + \
            ", ".join(c for c in ITEMS if X[c].iloc[0] and c.split("=")[0].endswith("_band"))
    return text, matched_rules(BEST, X)

pick = lambda c: gr.Dropdown(sorted(df[c].unique()), value=df[c].mode()[0], label=c)
demo = gr.Interface(
    fn=demo_predict,
    inputs=[gr.Number(30, label="Tuổi"), gr.Number(60000, label="Thu nhập năm (USD)"),
            gr.Number(10000, label="Số tiền vay (USD)"), gr.Number(11.0, label="Lãi suất (%)"),
            gr.Number(650, label="Điểm tín dụng"), pick("person_gender"), pick("person_education"),
            pick("person_home_ownership"), pick("loan_intent"), pick("previous_loan_defaults_on_file"),
            gr.Number(3, label="Số năm kinh nghiệm làm việc"), gr.Number(3, label="Độ dài lịch sử tín dụng (năm)")],
    outputs=[gr.Markdown(label="Kết quả"), gr.Dataframe(label="Luật khớp (tối đa 10)")],
    examples=[
        [35, 120000, 8000, 10.5, 700, "male", "Bachelor", "MORTGAGE", "EDUCATION", "No", 10, 8],
        [30, 60000, 10000, 11.0, 720, "female", "Master", "RENT", "PERSONAL", "Yes", 5, 6],
        [24, 20000, 9000, 16.5, 580, "female", "High School", "RENT", "MEDICAL", "No", 1, 2],
        [25, 45000, 9000, 12.0, 600, "male", "Bachelor", "RENT", "EDUCATION", "No", 3, 2],
    ],
    title="XÉT DUYỆT KHOẢN VAY ONLINE",
    description="Nhập hồ sơ → xem 3 thuật toán dự đoán và các luật khớp. Tỉ lệ vay/thu nhập được tự tính.",
)
demo.launch(share=True)
