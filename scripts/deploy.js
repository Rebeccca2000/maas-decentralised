// scripts/deploy.js  (Ethers v5)
const fs = require("fs");
const path = require("path");
const { ethers } = require("hardhat");

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deployer:", deployer.address);

  // 1) Deploy Registry
  const Registry = await ethers.getContractFactory("MaaSRegistry");
  const registry = await Registry.deploy();
  await registry.deployed();

  // 2) Deploy Request
  const Request = await ethers.getContractFactory("MaaSRequest");
  const request = await Request.deploy(registry.address);
  await request.deployed();

  // 3) Deploy Auction
  const Auction = await ethers.getContractFactory("MaaSAuction");
  const auction = await Auction.deploy(registry.address, request.address);
  await auction.deployed();

  // 4) Deploy Facade
  const Facade = await ethers.getContractFactory("MaaSFacade");
  const facade = await Facade.deploy(registry.address, request.address, auction.address);
  await facade.deployed();

  // 5) WIRE DIRECTLY AS OWNER (KISS)
  await (await request.setMarketplaceAddress(facade.address)).wait();
  await (await auction.setMarketplaceAddress(facade.address)).wait();

  // 6) Set marketplace API on Facade (owner-only)
  await (await facade.setMarketplaceAPI(deployer.address)).wait();

  // 7) Save addresses
  const out = {
    network: (await ethers.provider.getNetwork()).name || "localhost",
    Registry: registry.address,
    Request: request.address,
    Auction: auction.address,
    Facade: facade.address,
    marketplaceAPI: await facade.marketplaceAPI(),
    deployedBy: deployer.address,
    deployedAt: new Date().toISOString()
  };
  const dir = path.resolve("./deployed");
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "simplified.json"), JSON.stringify(out, null, 2));
  console.log("Deployed:", out);
}

main().catch((e) => { console.error(e); process.exit(1); });
