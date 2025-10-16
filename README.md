Api que converte imagens para diversos metodos de escala de cinza.

🎨 Técnicas disponíveis na API
| Técnica (`technique=`)           | Descrição                                                                                                                                                 | Exemplo de uso no Postman / URL           |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `average` (ou `avg`)             | Faz a média simples dos três canais (R, G, B). Cada cor tem o mesmo peso. <br>**Fórmula:** (R + G + B) / 3                                                | `?technique=average`                      |
| `luminosity`, `luma`, ou `bt709` | Usa **pesos perceptuais BT.709**, que consideram a sensibilidade do olho humano (mais sensível ao verde). <br>**Fórmula:** 0.2126 R + 0.7152 G + 0.0722 B | `?technique=luminosity`                   |
| `lightness`                      | Usa o valor **médio entre o máximo e o mínimo** dos canais. <br>**Fórmula:** (max(R,G,B) + min(R,G,B)) / 2                                                | `?technique=lightness`                    |
| `desaturation`                   | Idêntica à anterior (`lightness`), só muda o nome. Também faz (máximo + mínimo) / 2.                                                                      | `?technique=desaturation`                 |
| `single_channel`                 | Mantém **apenas um canal de cor** (vermelho, verde ou azul) e descarta os outros. <br>Você deve informar qual canal: `r`, `g`, ou `b`.                    | `?technique=single_channel&channel=r`     |
| `weighted`                       | Permite **definir manualmente os pesos** de cada canal (R, G, B). <br>Você passa um parâmetro `weights` no formato `wr,wg,wb`.                            | `?technique=weighted&weights=0.3,0.5,0.2` |
