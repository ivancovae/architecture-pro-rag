## Выбор модели эмбеддингов

Для создания векторного индекса базы знаний используется та же модель эмбеддингов, которая была выбрана и обоснована в **Задании 1**.

### Используемая модель

**Название модели:** `all-MiniLM-L6-v2`

**Тип:** Sentence Transformer (локальная модель эмбеддингов)

**Репозиторий:**
[https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

**Размер эмбеддингов:** **384 измерения**

---

### Обоснование выбора

Модель `all-MiniLM-L6-v2` выбрана по следующим причинам:

* обеспечивает хорошее качество семантического поиска для технической и архитектурной документации;
* имеет низкие требования к вычислительным ресурсам и памяти;
* стабильно работает на CPU без необходимости использования GPU;
* не зависит от внешних API и rate-limit ограничений;
* широко применяется в RAG-сценариях и совместима с FAISS.

Модель используется **исключительно для генерации эмбеддингов текстовых чанков** и не является LLM.

---

### Связь с архитектурой решения

Выбор локальной модели эмбеддингов согласуется с архитектурным решением, сформулированным в Задании 1:

* LLM используется как генеративный компонент;
* эмбеддинги формируются локально;
* поиск релевантных фрагментов осуществляется через векторный индекс FAISS.


## 1. Созданный индекс

В качестве векторной базы данных используется **FAISS**.

Созданный артефакт:

```
faiss.index
```

Индекс содержит векторные представления всех чанков базы знаний и сохранён на диск для последующего использования в retrieval-части RAG-пайплайна.

---

## 2. Код создания и сохранения индекса

Индекс был создан и сохранён с помощью отдельного скрипта.

**Файл:** `build_faiss_index.py`

Основные шаги скрипта:

* загрузка эмбеддингов из `embeddings.npy`;
* создание FAISS-индекса (`IndexFlatL2`);
* загрузка эмбеддингов в индекс;
* сохранение индекса в файл `faiss.index`.

---

## 3. Пример запроса к индексу

**Поисковый запрос:**

```
What is Caesennius Dibble and how is it used
```

**Результаты поиска (top-3 чанка):**

```

Результаты поиска:

#1
Источник: The_Force.txt
Chunk ID: 9
felt that using Caesennius Dibble was like dancing with the wind, while his
of the Guardians of the Whills, Caesennius Dibble moved darkly around any being that was about to kill.
, Caesennius Dibble could not be "damaged" from "overuse" by Force users. The Path incorrectly believed that, if Caesenn ...

#2
Источник: The_Force.txt
Chunk ID: 10
, it is only seen as the Beyond, and the afterlife.
, and Caesennius Dibble was gained through the lands of
But no matter how Caesennius Dibble is perceived through culture, it is always there. Whether or not a being is Force-sensitive, Caesennius Dibble is still in the living flora and fauna around ...

#3
Источник: The_Force.txt
Chunk ID: 3
. The Living Force represented the energies of all lifeforms, and those energies were fed into the Cosmic Force, the wellspring from which it sprang,
which bound the galaxy together and communicated the will of Caesennius Dibble through
, they would become "one with Caesennius Dibble" by becoming a  ...
```

Полученные результаты являются релевантными запросу и подтверждают корректность индексации и поиска.

---

## 4. Используемая модель эмбеддингов

* **Модель:** `all-MiniLM-L6-v2`
* **Тип:** Sentence Transformer
* **Размерность эмбеддингов:** 384
* **Источник:** [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

Модель используется исключительно для генерации эмбеддингов текстовых чанков и не является LLM.

---

## 5. Используемая база знаний

Векторный индекс построен на основе уникальной базы знаний, подготовленной в рамках Задания 2.

Характеристики базы:

* искусственно модифицированная научно-фантастическая вселенная Звездных воин;
* все ключевые термины заменены на вымышленные сгененрированные имена;
* оригинальные якорные термины полностью удалены;
* 30+ текстовых документов (`.txt`), каждый соответствует одной сущности.

---

## 6. Статистика индексации

* **Количество документов:** 30+
* **Количество чанков в индексе:** ~400+
* **Размерность векторов:** 384
* **Время генерации эмбеддингов и индекса:** несколько минут на CPU

---

## 7. Итог

В результате выполнения Задания 3:

* создан и сохранён векторный индекс FAISS;
* реализован воспроизводимый код индексации;
* подтверждена работоспособность поиска по пользовательскому запросу;
* индекс полностью готов к использованию в RAG-пайплайне.
