//===== funções auxiliares ====================

            function transpoe(a) {
                return a[0].map((x,i) => a.map((y) => y[i]));
            }

            function somaMatrizEscalar(m, e) {
                return m.map((v) => v.map((x) => parseFloat(x) + parseFloat(e)));
            }

            function subtraiEscalarMatriz(e, m) {
                return m.map((v) => v.map((x) => parseFloat(e) - parseFloat(x)));
            }

            function somaMatriz(m1, m2) {
                return m1.map((v1, linha) => v1.map((x, coluna) => parseFloat(m1[linha][coluna]) + parseFloat(m2[linha][coluna])));
            }

            function subtraiMatriz(m1, m2) {
                return m1.map((v1, linha) => v1.map((x, coluna) => parseFloat(m1[linha][coluna]) - parseFloat(m2[linha][coluna])));
            }

            function produtoEscalarVetor(a, b) {
                return [a.map((x,index) => a[index] * b[index]).reduce((m,n) => m + n)];
            }
        
            function produtoEscalar(a, b) {
                return a.map((x) => transpoe(b).map((y) => produtoEscalarVetor(x, y)));
            }

            function multiplicaMatriz(m1, m2) {
                return m1.map((v1, linha) => v1.map((x, coluna) => parseFloat(m1[linha][coluna]) * parseFloat(m2[linha][coluna])));
            }
            
            function sigmoid(m) {
                return m.map((v) => v.map((x) => 1 / (1 + Math.pow(Math.E, -x))));
            }

            function derivadaSigmoid(a) {
                return multiplicaMatriz(a, subtraiEscalarMatriz(1, a));
            }

            function geraMatrizPesos(numLinhas, numColunas) {
                let pesos = [];
                for(let i = 0; i < numLinhas; i++) {
                    let linha = [];
                    for(let j = 0; j < numColunas; j++) {
                        let peso = Math.random();
                        linha.push(peso);
                    }
                    pesos.push(linha);
                }
                return pesos;
            }

            //==============================================

            let entradas = [
                [0,0],
                [0,1],
                [1,0],
                [1,1]
            ];

            let y_esperado = transpoe([[0, 1, 1, 0]]);

            let pesosEntrada = geraMatrizPesos(2,16);            
            let pesosCamadaEscondida = geraMatrizPesos(16,1);
            
            let y = null;
           
            for(let i = 0; i < 10000; i++) {
                let z = sigmoid(produtoEscalar(entradas, pesosEntrada));
                y = sigmoid(produtoEscalar(z, pesosCamadaEscondida));
               
                let erro = subtraiMatriz(y, y_esperado);
                
                let derivadaY = derivadaSigmoid(y);
                let derivadaZ = derivadaSigmoid(z);
                
                let deltaPesosEscondida = produtoEscalar(transpoe(z), multiplicaMatriz(erro, derivadaY));
                let deltaPesosEntrada = produtoEscalar(transpoe(entradas), multiplicaMatriz(produtoEscalar(multiplicaMatriz(erro, derivadaY), transpoe(pesosCamadaEscondida)), derivadaZ));
              
                pesosCamadaEscondida = subtraiMatriz(pesosCamadaEscondida, deltaPesosEscondida);
                pesosEntrada = subtraiMatriz(pesosEntrada, deltaPesosEntrada);
            }

            console.log("Saída após o treinamento: ", y);
