# Table Layout Description

## Overview

The target documents contain **semi-structured financial/accounting tables** that must be reconstructed into a logically correct HTML table rather than merely reproducing the visual appearance.

Unlike conventional table recognition benchmarks, these tables contain a combination of:

- Hierarchical headers
- Sparse body cells
- Implicit row grouping
- Variable logical subcolumns
- Large empty regions
- Weak horizontal separators
- Strong semantic relationships between neighboring columns

The goal is **logical reconstruction**, where each printed text fragment is assigned to its correct logical cell.

---

# Overall Layout

The table is composed of two major parts:

1. Header
2. Body

The header remains almost identical across documents, while the body varies.

---

# Header Structure

The header consists of **two levels**.

Example:

```
+-----------------------------------------------------------+
| Currency | Figure100 | Figures                            |
|          |           | Debit | Credit                     |
+-----------------------------------------------------------+
```

The first header row contains **parent headers**.

Typical parent headers include:

- Currency
- Figure100
- Figures

These parent headers span multiple logical columns.

---

## Parent Header: Currency

Currency logically represents **two child columns**.

Example:

| Description | Value |
|------------|------|
| Return | 3% |
| Claim | 1000 |

Although these child headers are usually **not printed**, they are implied by the body.

Logical structure:

```
Currency
├── Description
└── Value
```

---

## Parent Header: Figure100

Figure100 is the most difficult section.

It normally contains:

```
Figure100
├── Number
├── Type
└── Optional
```

where

- Number is numeric
- Type usually contains symbols such as D or C
- Optional may contain values such as Nil or other document-specific content

However,

the Optional column **does not always exist**.

Therefore Figure100 may logically contain

```
Number
Type
```

or

```
Number
Type
Optional
```

depending on the body.

Importantly,

the printed header does **not** indicate whether the Optional column exists.

The model must infer it from the body.

---

## Parent Header: Figures

Figures usually contains

```
Figures
├── Debit
└── Credit
```

These two logical columns are generally stable.

---

# Body Structure

The body contains several logical record groups.

Example:

```
ABC

Return
Claim

DEF

Return
Claim
```

The group name (ABC, DEF, etc.) appears in the first logical column.

It acts as a section identifier rather than a standard data row.

---

# Logical Rows

Within each group,

rows typically represent individual financial items.

Example

| Description | Value |
|------------|------|
| Return | 3% |
| Claim | 1000 |

Each logical row extends across every logical column of the table.

---

# Sparse Cells

Many logical cells are intentionally empty.

For example

| Description | Value | Number | Type | Optional | Debit | Credit |
|------------|------|--------|------|----------|------|-------|
| Return | 3% | 123 | D | | | 567 |
| Claim | 1000 | 234 | C | | | |

Empty cells must be preserved.

The model must never shift neighboring values to fill empty positions.

---

# Variable Logical Schema

This is the most important characteristic.

The number of logical columns cannot always be determined from the header.

Example 1

```
Figure100

123
D
Nil
```

Logical interpretation

```
Number
Type
Optional
```

Example 2

```
Figure100

123
D
```

Logical interpretation

```
Number
Type
```

Although the header remains identical,

the logical schema differs.

Therefore,

the body must be analyzed jointly with the header.

---

# Implicit Columns

Some logical columns have no printed child header.

Instead,

their existence is inferred from

- vertical alignment
- semantic consistency
- neighboring values

This differs from conventional table datasets where every logical column is explicitly labeled.

---

# Implicit Row Boundaries

Horizontal rules are often sparse.

Logical rows are determined primarily by

- text alignment
- spacing
- semantic consistency

rather than explicit grid lines.

---

# Semantic Consistency

Column assignment depends not only on geometry but also on value semantics.

Typical examples

Number column

```
123
456
789
```

Type column

```
D
C
```

Optional column

```
Nil
```

Value column

```
3%
1000
2500
```

Debit / Credit

```
567
891
```

Semantics should be used together with visual alignment.

---

# Empty Cells

Blank cells represent valid table content.

Blank does NOT imply

- missing OCR
- missing detection
- merge with neighboring cell

Blank cells should remain blank.

---

# Section Rows

Rows such as

```
ABC
DEF
XYZ
```

serve as group headers.

They introduce a new record block.

These rows usually contain text only in the first logical column.

Remaining logical columns are empty.

---

# Summary Rows

Rows such as

```
Total
Sum
Final
```

may appear near the bottom.

These are ordinary logical rows whose values may exist only in selected columns.

---

# Merged Cells

Merged cells may occur in

- parent headers
- section rows
- summary rows

Both rowspan and colspan should be preserved whenever required.

---

# Logical Reconstruction Objective

The objective is NOT to reproduce the visual layout.

Instead,

the objective is to recover the latent logical table.

The reconstructed HTML should satisfy

- correct logical rows
- correct logical columns
- correct hierarchical headers
- correct rowspan
- correct colspan
- correct placement of every printed text fragment

---

# Comparison with Conventional Table Recognition

Conventional TSR assumes

```
Image
↓

Known table schema
↓

Recover cells
```

These documents require

```
Image
↓

Infer logical schema

↓

Infer logical rows

↓

Infer logical columns

↓

Assign every text fragment

↓

Generate HTML
```

Schema inference is therefore an essential component of the task.

---

# Core Challenges

The primary challenges include

1. Hierarchical headers

2. Implicit child columns

3. Variable logical schema

4. Sparse body cells

5. Weak horizontal separators

6. Implicit row boundaries

7. Grouped records

8. Semantic column inference

9. Correct handling of blank cells

10. Faithful HTML reconstruction

---

# Characteristics of the Dataset

The tables can be characterized as

- Semi-structured
- Financial / accounting style
- Hierarchical
- Sparse
- Layout-driven
- Semantically constrained
- Variable-schema
- Document-oriented

rather than regular spreadsheet-like tables.

---

# Reconstruction Principles

A reconstruction system should follow these principles:

1. Preserve every visible printed value.

2. Never hallucinate missing values.

3. Never merge adjacent logical columns simply because one column is sparse.

4. Infer optional columns only when supported by body evidence.

5. Use both geometry and semantic consistency during column assignment.

6. Preserve logical row grouping.

7. Produce structurally valid HTML.

8. Favor logical correctness over visual similarity.

9. Keep blank cells blank.

10. Maintain parent-child header relationships.