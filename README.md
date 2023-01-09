# Blockchain based Cloud Gaming Streamer

The goal of this project is to construct a distributed setup to help gamers connect to each other.  
The gamer-to-gamer (G2G) concept is like a cloud gaming network but gamers provide their own setup.  
Transactions are done automatically via smart contracts and the network is scalable.

## Blockchain

I implemented the blockchain infrastructure of the entire network from scratch. This gives me more 
flexibility to scale the network according to the network's needs.

## Transformer

For the communication I used a Variational Auto-Encoder neural network based on transformers.  
This model will be trained on huge video datasets and can compress and decompress the stream 
of data. This reduces the load on network and makes the latency smaller.

## P2P

Here, we do everything from initiating connection to sending streams and closing the connection 
after completion. Users are connected P2P, their records are entered into the blockchain and 
connection is maintained by the network.

### This project is still ongoing and many functionalities are still unavailable
