// File: scripts/deploy.js
const { ethers } = require("hardhat");

async function main() {
  console.log("Starting deployment of MaaS smart contracts...");

  // Get the signers
  const [deployer] = await ethers.getSigners();
  console.log(`Deploying contracts with the account: ${deployer.address}`);

  // Deploy MaaSRegistry first
  console.log("Deploying MaaSRegistry...");
  const MaaSRegistry = await ethers.getContractFactory("MaaSRegistry");
  const registry = await MaaSRegistry.deploy();
  await registry.deployed();
  console.log(`MaaSRegistry deployed to: ${registry.address}`);

  // Deploy MaaSRequest
  console.log("Deploying MaaSRequest...");
  const MaaSRequest = await ethers.getContractFactory("MaaSRequest");
  const request = await MaaSRequest.deploy(registry.address);
  await request.deployed();
  console.log(`MaaSRequest deployed to: ${request.address}`);

  // Deploy MockERC20 (for payments)
  console.log("Deploying MockERC20...");
  const MockERC20 = await ethers.getContractFactory("MockERC20");
  const mockToken = await MockERC20.deploy("MaaS Token", "MAAS", ethers.utils.parseEther("1000000"));
  await mockToken.deployed();
  console.log(`MockERC20 deployed to: ${mockToken.address}`);

  // Deploy MaaSAuction
  console.log("Deploying MaaSAuction...");
  const MaaSAuction = await ethers.getContractFactory("MaaSAuction");
  const auction = await MaaSAuction.deploy(registry.address, request.address);
  await auction.deployed();
  console.log(`MaaSAuction deployed to: ${auction.address}`);

  // Deploy MaaSNFT
  console.log("Deploying MaaSNFT...");
  const MaaSNFT = await ethers.getContractFactory("MaaSNFT");
  const nft = await MaaSNFT.deploy(registry.address, request.address, auction.address, mockToken.address);
  await nft.deployed();
  console.log(`MaaSNFT deployed to: ${nft.address}`);

  // Deploy MaaSMarket
  console.log("Deploying MaaSMarket...");
  const MaaSMarket = await ethers.getContractFactory("MaaSMarket");
  const market = await MaaSMarket.deploy(nft.address, mockToken.address);
  await market.deployed();
  console.log(`MaaSMarket deployed to: ${market.address}`);

  // Deploy MaaSFacade
  console.log("Deploying MaaSFacade...");
  const MaaSFacade = await ethers.getContractFactory("MaaSFacade");
  const facade = await MaaSFacade.deploy(
    registry.address,
    request.address,
    auction.address,
    nft.address,
    market.address
  );
  await facade.deployed();
  console.log(`MaaSFacade deployed to: ${facade.address}`);

  // Save deployment information to a JSON file
  const fs = require("fs");
  const deploymentInfo = {
    registry: registry.address,
    request: request.address,
    auction: auction.address,
    nft: nft.address,
    market: market.address,
    facade: facade.address,
    mockToken: mockToken.address,
    network: network.name,
    deployer: deployer.address,
    timestamp: Date.now()
  };

  fs.writeFileSync(
    "deployment-info.json",
    JSON.stringify(deploymentInfo, null, 2)
  );
  console.log("Deployment information saved to deployment-info.json");

  console.log("MaaS contract deployment completed successfully!");
}

// Execute the deployment
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Error during deployment:", error);
    process.exit(1);
  });