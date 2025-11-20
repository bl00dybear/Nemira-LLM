## 1. Continued Pre-training (Domain Adaptation)

```py
# Scopul: Modelul să învețe vocabular și stil specific FMI
# Exemplu:
texts = [
    "Facultatea de Matematică și Informatică organizează admiterea în iulie...",
    "Biblioteca FMI dispune de resurse în matematică, informatică și statistică...",
    "Secretariatul studenților se ocupă de procesarea documentelor academice..."
]

# Modelul învață:
# - Termeni specifici: "FMI", "admitere", "secretariat", "bibliotecă"
# - Structuri de fraze din domeniu
# - Context despre universitate
```

## 2. Instruction Fine-tuning (Q&A Format)

### Format Mistral-Instruct JSONL

```py
sample_qa_dataset = [
    {
        "messages": [
            {
                "role": "user",
                "content": "Când sunt admiterile la FMI?"
            },
            {
                "role": "assistant",
                "content": "Admiterile la Facultatea de Matematică și Informatică se desfășoară în perioada iulie-septembrie. Sesiunea de vară este în iulie, iar sesiunea de toamnă în septembrie. Datele exacte sunt anunțate pe site-ul oficial fmi.unibuc.ro."
            }
        ]
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Câte locuri sunt la buget la Informatică?"
            },
            {
                "role": "assistant",
                "content": "La secția Informatică sunt aproximativ 300 de locuri la buget anual, distribuite astfel: ~240 locuri pentru Informatică în limba română și ~60 locuri pentru Informatică în limba engleză. Numărul exact variază ușor de la an la an."
            }
        ]
    },

]
```