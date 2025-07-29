// File: scripts/test-maas-flow.js
const { ethers } = require("hardhat");
const fs = require("fs");

// Load deployment info
const deploymentInfo = JSON.parse(fs.readFileSync("deployment-info.json", "utf8"));

async function main() {
  console.log("Starting MaaS flow test...");

  // Get the signers
  const [admin, commuter1, commuter2, provider1, provider2] = await ethers.getSigners();
  
  console.log(`Admin: ${admin.address}`);
  console.log(`Commuter 1: ${commuter1.address}`);
  console.log(`Commuter 2: ${commuter2.address}`);
  console.log(`Provider 1: ${provider1.address}`);
  console.log(`Provider 2: ${provider2.address}`);

  // Load the contracts
  const registry = await ethers.getContractAt("MaaSRegistry", deploymentInfo.registry);
  const request = await ethers.getContractAt("MaaSRequest", deploymentInfo.request);
  const auction = await ethers.getContractAt("MaaSAuction", deploymentInfo.auction);
  const nft = await ethers.getContractAt("MaaSNFT", deploymentInfo.nft);
  const market = await ethers.getContractAt("MaaSMarket", deploymentInfo.market);
  const mockToken = await ethers.getContractAt("MockERC20", deploymentInfo.mockToken);
  const facade = await ethers.getContractAt("MaaSFacade", deploymentInfo.facade);

  // Distribute tokens to all participants
  console.log("Distributing tokens to participants...");
  await mockToken.transfer(commuter1.address, ethers.utils.parseEther("1000"));
  await mockToken.transfer(commuter2.address, ethers.utils.parseEther("1000"));
  await mockToken.transfer(provider1.address, ethers.utils.parseEther("1000"));
  await mockToken.transfer(provider2.address, ethers.utils.parseEther("1000"));

  // Register commuters
  console.log("Registering commuters...");
  await registerCommuter(registry, commuter1, 1);
  await registerCommuter(registry, commuter2, 2);
  
  // Verify commuter registration was successful
  const commuter1Info = await registry.getCommuter(commuter1.address);
  console.log(`Commuter1's registered ID: ${commuter1Info[0]}`);
  
  // Register providers
  console.log("Registering providers...");
  await registerProvider(registry, provider1, 1, "UberLike1", 3); // car
  await registerProvider(registry, provider2, 2, "BikeShare1", 4); // bike

  const provider1Info = await registry.getProvider(provider1.address);
  console.log(`Provider1's registered ID: ${provider1Info.providerId}`);

  // Step 1: Create travel request directly through the Request contract for more control
  console.log("Creating travel request directly...");
  const requestId = 1;
  const origin = [1, 1];
  const destination = [10, 10];
  const startTime = Math.floor(Date.now() / 1000) + 3600; // 1 hour from now
  const travelPurpose = 0; // Work
  const flexibleTime = "30 minutes";
  const requirementValues = [false, false, false, false]; // No special requirements
  const requirementKeys = ["wheelchair", "assistance", "child_seat", "pet_friendly"];
  const pathInfo = JSON.stringify({
    route: [
      [1, 1],
      [5, 5],
      [10, 10]
    ],
    distance: 15,
    estimatedTime: 30
  });

  await facade.connect(commuter1).createTravelRequest(
    requestId,
    origin,
    destination,
    startTime,
    travelPurpose,
    flexibleTime,
    requirementValues,
    requirementKeys,
    pathInfo
  );
    
  console.log(`Travel request ${requestId} created directly`);

  // Verify request creation
  const requestDetails = await request.getRequestBasicInfo(requestId);
  console.log(`Request's commuterId: ${requestDetails[1]}`);
  console.log(`Request's status: ${requestDetails[6]}`); // Status should be Active (0)
  
  // Now create an auction for the request
  console.log("Creating auction for request...");

  
  await auction.connect(admin).createAuction(requestId, pathInfo);
  console.log(`Auction created for request ${requestId}`);

  // Step 3: Submit offers
  console.log("Submitting offers...");
  await submitOffer(auction, provider1, requestId, ethers.utils.parseEther("25"));
  await submitOffer(auction, provider2, requestId, ethers.utils.parseEther("15"));

  // Step 4: Finalize auction
  console.log("Finalizing auction...");
  await auction.connect(admin).finalizeAuction(requestId);
  
  // Get winning offer to identify the provider
  const winningOffers = await auction.getWinningOffers(requestId);
  const winningProviderId = winningOffers[0].providerId;
  const winningPrice = winningOffers[0].price;
  console.log(`Winning provider ID: ${winningProviderId} with price ${ethers.utils.formatEther(winningPrice)} tokens`);

  // Step 5: Let's verify the request ownership before proceeding
  const requestOwner = await request.connect(commuter1).getRequestDetails(requestId);
  console.log(`Request owner address: ${requestOwner.owner}`);
  console.log(`Commuter1 address: ${commuter1.address}`);
  
  // Check if the commuter ID matches
  console.log(`Request commuterId: ${requestOwner.commuterId}`);
  console.log(`Commuter1's ID from registry: ${commuter1Info[0]}`);
  
  // UPDATED: Use admin to update the request status first
  console.log("Updating request status using admin...");
  await request.connect(admin).updateRequestStatus(requestId, 3); // 3 = ServiceSelected
  
  // Approve tokens for potential payment to provider
  console.log("Approving tokens for provider payment...");
  const providerAddress = await registry.providerIdToAddress(winningProviderId);
  await mockToken.connect(commuter1).approve(providerAddress, winningPrice);
  
  // Transfer payment to provider directly 
  console.log("Transferring payment to provider directly...");
  await mockToken.connect(commuter1).transfer(providerAddress, winningPrice);
  
  // Prepare service details
  const routeDetails = JSON.stringify({
    route: [
      [1, 1],
      [5, 5],
      [10, 10]
    ],
    distance: 15,
    estimatedTime: 30,
    completed: true
  });
  
  const serviceDuration = 30 * 60; // 30 minutes in seconds
  const tokenURI = "ipfs://QmServiceMetadata";
  
  // Mint the NFT directly
  console.log("Minting service NFT directly...");
  const tx = await nft.connect(commuter1).mintServiceNFT(
    requestId,
    winningProviderId,
    routeDetails,
    winningPrice,
    startTime,
    serviceDuration,
    tokenURI
  );
  
  const receipt = await tx.wait();
  // Find the ServiceTokenized event
  const event = receipt.events.find(e => e.event === 'ServiceTokenized');
  const tokenId = event.args.tokenId;
  
  console.log(`Service agreement completed for request ${requestId}, NFT minted with token ID: ${tokenId}`);

  // Step 6: List NFT for sale
  console.log("Listing NFT for sale...");
  const salePrice = ethers.utils.parseEther("20");
  
  // Approve market to transfer the NFT
  await nft.connect(commuter1).approve(market.address, tokenId);
  
  // List on the market
  await market.connect(commuter1).listNFTForSale(tokenId, salePrice);
  console.log(`NFT ${tokenId} listed for sale at ${ethers.utils.formatEther(salePrice)} tokens`);

  // Step 7: Purchase NFT
  console.log("Purchasing NFT...");
  await mockToken.connect(commuter2).approve(market.address, salePrice);
  await market.connect(commuter2).purchaseNFT(tokenId);
  console.log(`NFT ${tokenId} purchased successfully by Commuter2`);

  console.log("MaaS flow test completed successfully!");
}

// The rest of the functions remain unchanged
async function registerCommuter(registry, signer, commuterId) {
  // First check if the commuter is already registered with a different ID
  try {
    const existingCommuter = await registry.getCommuter(signer.address);
    if (existingCommuter[0].toNumber() > 0) {
      console.log(`Commuter already registered with ID: ${existingCommuter[0]}`);
      return;
    }
  } catch (error) {
    console.log("Commuter not registered yet, will register now");
  }
  
  const commuterData = {
    commuterId: commuterId,
    location: [1, 1], // x, y coordinates
    incomeLevel: 1, // middle income (0: low, 1: middle, 2: high)
    preferredMode: 2, // car (enum value)
    age: 35,
    hasDisability: false,
    techAccess: true,
    healthStatus: 0, // good health (0: good, 1: poor)
    paymentScheme: 0 // PAYG (0: PAYG, 1: Subscription)
  };

  await registry.connect(signer).addCommuter(
    commuterData.commuterId,
    commuterData.location,
    commuterData.incomeLevel,
    commuterData.preferredMode,
    commuterData.age,
    commuterData.hasDisability,
    commuterData.techAccess,
    commuterData.healthStatus,
    commuterData.paymentScheme
  );
  
  // Verify registration succeeded
  const registeredCommuter = await registry.getCommuter(signer.address);
  console.log(`Commuter ${commuterId} registered successfully with ID: ${registeredCommuter[0]}`);
}

async function registerProvider(registry, signer, providerId, companyName, modeType) {
    const providerData = {
      providerId: providerId,
      providerAddress: signer.address,
      companyName: companyName,
      modeType: modeType,
      basePrice: ethers.utils.parseEther("10"),
      capacity: 50,
      // Enhanced provider attributes
      registrationTime: 0, // Will be set by contract
      serviceCount: 0,
      totalRevenue: 0,
      availableCapacity: 50,
      servicingArea: 10,
      serviceCenter: [5, 5], // Location coordinates
      responseTime: 5,
      reliability: 70,
      qualityScore: 70,
      certifications: [],
      isVerified: false,
      isActive: true
    };
  
    await registry.connect(signer).addProvider(providerId, providerData);
    console.log(`Provider ${providerId} (${companyName}) registered successfully`);
}

async function submitOffer(auction, signer, requestId, price) {
  const auctionId = requestId; // Assuming 1:1 mapping for simplicity
  
  // Get the provider ID by direct query to the registry contract
  // First get the registry address
  const registryAddress = await auction.registry();
  const registry = await ethers.getContractAt("MaaSRegistry", registryAddress);
  const provider = await registry.getProvider(signer.address);
  
  const providerId = provider.providerId;
  const modeType = provider.modeType;
  
  console.log(`Submitting offer as provider ${providerId} with mode ${modeType}`);
  
  const offerData = {
    id: 0, // Will be set by contract
    requestId: requestId,
    providerId: providerId,
    auctionId: auctionId,
    price: price,
    mode: modeType,
    startTime: Math.floor(Date.now() / 1000) + 3600, // 1 hour from now
    totalTime: 30 * 60, // 30 minutes in seconds
    totalPrice: price,
    routeDetails: JSON.stringify({
      route: [
        [1, 1],
        [5, 5],
        [10, 10]
      ],
      estimatedTime: 30,
      distance: 15
    }),
    // Enhanced offer attributes
    offerTime: 0, // Will be set by contract
    capacity: 10,
    quality: 80,
    reliability: 85,
    isVerified: true,
    hasInsurance: true,
    additionalServices: [],
    isConditional: false,
    conditions: ""
  };

  const tx = await auction.connect(signer).submitOffer(offerData);
  await tx.wait();
  console.log(`Provider ${providerId} submitted offer for request ${requestId} at price ${ethers.utils.formatEther(price)} tokens`);
}

// Execute the test
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Error during test:", error);
    process.exit(1);
  });