Is = 0.01e-12; Ib = 0.1e-12; Vb = 1.3; Gp = 0.1;
V = linspace(-1.95,0.7,200);

I = Is*(exp(1.2/0.025*V)-1) + Gp*V - Ib*(exp(-1.2/0.025*(V+Vb))-1);
II = I.*(rand(1,200)*0.4+0.8);

fit1 = polyfit(V,I,4);
fit2 = polyfit(V,I,8);
y1 = polyval(fit1,V);
y2 = polyval(fit2,V);

subplot(2,2,1), plot(V,I,V,y1,V,y2,V,II,'g')
legend('polyfitted','4th order','8th order','I +- 20%')
title('Polynomial fit comparisons')
xlabel('Voltage')
ylabel('Current')
subplot(2,2,2), semilogy(V,abs(I),'b',V,abs(II),'g')
legend('I','I +- 20%')
title('Logplot of I and II')
xlabel('Voltage')
ylabel('Current')

V = reshape(V,[200,1]);
I = reshape(I,[200,1]);

fo = fittype('A.*(exp(1.2*x/25e-3)-1) + Gp.*x - C*(exp(1.2*(-(x+Vb))/25e-3)-1)');
ff = fit(V,I,fo);
If = ff(V);

fo1 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+Vb))/25e-3)-1)');
ff1 = fit(V,I,fo1);
If1 = ff1(V);

fo2 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
ff2 = fit(V,I,fo2);
If2 = ff2(V);

If = reshape(If,[1,200]);
If1 = reshape(If1,[1,200]);
If2 = reshape(If2,[1,200]);
V = reshape(V,[1,200]);
I = reshape(I,[1,200]);

subplot(2,2,3), plot(V,I,V,If,V,If1,V,If2,'g')
legend('Original','A,C','A,B,C','A,B,C,D')

inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
view(net)
Inn = outputs

subplot(2,2,4), plot(V,Inn)
