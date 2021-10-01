# Wave Energy Converters
# Machine Learning
Nuestro problema gira entorno a la obtención de energı́a renovables a través de las olas. El medio marino es una fuente de energı́a poco explotada y para la que se han ideado múltiples dispositivos para la obtención de energı́a renovable del mismo. Nuestro problema en concreto usa unos dispositivos WECs (Wave Energy Converters) denominados CETO.

Vamos a tratar con una base de datos que podemos encontrar en https://archive.ics.uci.edu/ml/datasets/Wave+Energy+Converters. Analizaremos el problema correspondiente a la base de datos y conforme a ello aplicaremos técnicas de Aprendizaje Automático para resolverlo. La base de datos contiene información sobre la posición y potencia absorbida por los WECs en cuatro escenarios distintos de la costa australiana: Sydney, Adelaide, Perth y Tasmania. Los WECs serán dispositivos CETO y habrá 16 colocados en un área restringida: 566m².

## Preprocesado
Especial interés en este trabajo tiene el preprocesado de los datos. Todo lo relativo a ello se puede leer en la memoria que viene en el repositorio.

## Resultados
Nuestro mejor modelo es Random Forest que consigue un error R² en test de 0.9449.
