# TOM RAPPERSON
Library for Russian rap generation.

## Quick Start
Start a container:
```
git clone https://github.com/alexeykarnachev/tom_rapperson.git && \
cd tom_rapperson && \
docker build -t tom_rapperson . && \
docker run --name tom_rapperson -d -it --gpus all --network host tom_rapperson
```

Enter the container and run an example:
```
docker exec -it tom_rapperson bash \
python example.py
```

It will produce verses...
```
Её локоны так манят, как в моём сердце бренькало!
У меня есть мечта – это брак с примирением
Я хочу быть дождём над морем цвета сиреневого
И я не знаю, что тебя убьёт в самогоноварении

Я не эмигрант, не иммигрировавший
С детства я усвоил центрирование –
Всю жизнь я был фрустрирующий
И всю жизнь меня мама администрировала

Я вижу твоё лицо, оно очень пристальное
Слышишь, я чувствую боль и у меня приступы
На моей руке татухи, они бля ребристые
Подарил этой суке кольцо с ирисками

Вы мне не поможете, вы все переженитесь!
Я как будто бы в фильме Стэнли Кубрика - дело в самоуважении
Моя жизнь - это драма, моя позиция пораженческая
Судьба свела нас с тобой, но я не знал твоего местерасположения
```
