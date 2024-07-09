    public KeyPairGenerator getKeyPairGen() throws Exception {
        KeyPairGenerator keyPairGen = KeyPairGenerator.getInstance("RSA");
        // initialize keyPairGen and return it
        keyPairGen.initialize(