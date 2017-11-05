package sample;

import javafx.scene.control.TextArea;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class cellbody {
    private Double min;
    private Integer showNumber;
    private TextArea textOutput;
    public void setMin(Double min) {
        this.min = min;
    }

    public void setShowNumber(Integer showNumber) {
        this.showNumber = showNumber;
    }

    public void setTextOutput(TextArea textOutput) {
        this.textOutput = textOutput;
    }

    public Map<Integer,Double> initW(){
        Map<Integer,Double> w = new HashMap<>();
        //double w1 = (Math.random()*(2-0) - 1);
        //double w2 = (Math.random()*(2-0) - 1);
        w.put(0,1.0);
        w.put(1,-1.0);
        w.put(2,0.0);
        w.put(3,0.5);

        return w;
    }
    public List<List<Double>> initX(){
        List<List<Double>> x = new ArrayList<>();
        List<Double> x1 = new ArrayList();
        x1.add(1.0);
        x1.add(-2.0);
        x1.add(0.0);
        x1.add(-1.0);


        List<Double> x2 = new ArrayList();
        x2.add(0.0);
        x2.add(1.5);
        x2.add(-0.5);
        x2.add(-1.0);

        List<Double> x3 = new ArrayList();
        x3.add(-1.0);
        x3.add(1.0);
        x3.add(0.5);
        x3.add(-1.0);

        x.add(x1);
        x.add(x2);
        x.add(x3);

        return x;
    }

    public Map<List<Double>,Integer> initXOut(List<List<Double>> x){
        Map<List<Double>,Integer> xOut = new HashMap<>();
        xOut.put(x.get(0),-1);
        xOut.put(x.get(1),-1);
        xOut.put(x.get(2),1);
        return xOut;
    }

    public  Map<Integer,Double> train(Map<Integer, Double> w, List<List<Double>> x, Map<List<Double>, Integer> xOut, double η){
        Double stop = 0.0;
        int stopcount = 1;
        int  runtimes = 0;
        DecimalFormat df=new DecimalFormat("#.###");
        do{
            int xprint = 1;
            //show出訓練結果
            if(runtimes%showNumber==0){
                textOutput.appendText("--------------第"+stopcount*showNumber+"代---------------\n");
            }
            runtimes++;
            stop = 0.0;

            for(List<Double> xi:x){
                int xiSize = xi.size();
                double wtx = 0;

                //計算出 y
                Double y = countY(xi,w);
                //計算出 a'(net)
                Double aNet = (1-Math.pow(y,2))/2;
                //計算出 w2
                Double ηyaNet = η*(xOut.get(xi)-y)*aNet;
                for(int i=0;i<xiSize;i++){
                    Double w2 = ηyaNet*xi.get(i)+w.get(i);
                    w.put(i,w2);
                }
                stop += Math.abs(xOut.get(xi) - y);


                if(runtimes%showNumber==0){
                    textOutput.appendText("w"+xprint+" : ");
                    for(int key:w.keySet()){
                        String Strw=df.format(w.get(key));
                        textOutput.appendText(Strw+" , ");
                    }
                    String ynew=df.format(y);
                    textOutput.appendText("Y&d: "+ynew+"&"+xOut.get(xi)+"\n");
                }
                xprint++;
            }
            if(runtimes%showNumber==0){
                textOutput.appendText("stop : "+(stop/3)+"\n");
                stopcount++;
            }
        }while ((stop/3)>min);
        textOutput.appendText("一共訓練了:"+runtimes+"代\n");
        return w;
    }

    private Double countY(List<Double> x, Map<Integer, Double> w){
        int xLength = x.size();
        //算出net : net = w反矩陣 * x矩陣
        Double net = 0.0;
        for(int i=0;i<xLength;i++){
            if(x.get(i)==0 || w.get(i)==0)
                continue;
            net += x.get(i)*w.get(i);
        }
        //算出y : y = a(net) = 2/(1+exp(-net))-1
        Double y = 2/(1+Math.exp(-net))-1;
        return y;
    }
}
