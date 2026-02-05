import java.util.Scanner;

interface paymentgateway {
    void pay(int amount);
    void refund(int amount);
}

class Paytm implements paymentgateway {
    public void pay(int amount) {
        System.out.println("Paid " + amount + " using Paytm");
    }
    public void refund(int amount) {
        System.out.println("Refunded " + amount + " using Paytm");
    }
}

class PhonePe implements paymentgateway {
    public void pay(int amount) {
        System.out.println("Paid " + amount + " using PhonePe");
    }
    public void refund(int amount) {
        System.out.println("Refunded " + amount + " using PhonePe");
    }
}

class interfaces {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter amount to pay:");
        int amount = sc.nextInt();

        paymentgateway pg;

        pg = new Paytm();     
        pg.pay(amount);
        pg.refund(amount);

        pg = new PhonePe();  
        pg.pay(amount);
        pg.refund(amount);
    }
}
