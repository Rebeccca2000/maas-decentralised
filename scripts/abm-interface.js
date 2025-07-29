// File: scripts/abm-interface.js
const fs = require('fs');
const { ethers } = require('hardhat');

// Load deployment info
const deploymentInfo = JSON.parse(fs.readFileSync("deployment-info.json", "utf8"));

/**
 * This script provides functions for integrating the ABM model with the blockchain contracts.
 * It handles the communication between the two systems.
 */
class BlockchainABMInterface {
  constructor() {
    this.loadContracts();
    this.accounts = {};
  }

  async loadContracts() {
    this.registry = await ethers.getContractAt("MaaSRegistry", deploymentInfo.registry);
    this.request = await ethers.getContractAt("MaaSRequest", deploymentInfo.request);
    this.auction = await ethers.getContractAt("MaaSAuction", deploymentInfo.auction);
    this.nft = await ethers.getContractAt("MaaSNFT", deploymentInfo.nft);
    this.market = await ethers.getContractAt("MaaSMarket", deploymentInfo.market);
    this.facade = await ethers.getContractAt("MaaSFacade", deploymentInfo.facade);
    this.mockToken = await ethers.getContractAt("MockERC20", deploymentInfo.mockToken);
    
    // Get the default admin signer
    const [admin] = await ethers.getSigners();
    this.admin = admin;
    console.log("Contracts loaded successfully");
  }

  /**
   * Register a commuter agent from the ABM model
   * @param {Object} commuterAgent - The commuter agent from the ABM model
   * @returns {string} - The account address
   */
  async registerCommuter(commuterAgent) {
    // Create a new wallet for this commuter if not already registered
    if (!this.accounts[commuterAgent.unique_id]) {
      const wallet = ethers.Wallet.createRandom().connect(ethers.provider);
      this.accounts[commuterAgent.unique_id] = wallet;
      
      // Fund the account with ETH for gas
      await this.admin.sendTransaction({
        to: wallet.address,
        value: ethers.utils.parseEther("0.1")
      });
      
      // Fund with mock tokens
      await this.mockToken.connect(this.admin).transfer(
        wallet.address, 
        ethers.utils.parseEther("1000")
      );
    }
    
    const wallet = this.accounts[commuterAgent.unique_id];
    
    // Map income level to enum
    const incomeLevelMap = {
      "low": 0, 
      "middle": 1, 
      "high": 2
    };
    
    // Map payment scheme to enum
    const paymentSchemeMap = {
      "PAYG": 0,
      "subscription": 1
    };
    
    // Map health status to enum
    const healthStatusMap = {
      "good": 0,
      "poor": 1
    };
    
    // Create commuter data object for blockchain
    const commuterData = {
      commuterId: commuterAgent.unique_id,
      location: commuterAgent.location,
      incomeLevel: incomeLevelMap[commuterAgent.income_level],
      preferredMode: 5, // Default to MaaS bundle
      age: commuterAgent.age,
      hasDisability: commuterAgent.has_disability,
      techAccess: commuterAgent.tech_access,
      healthStatus: healthStatusMap[commuterAgent.health_status],
      paymentScheme: paymentSchemeMap[commuterAgent.payment_scheme]
    };
    
    // Register on blockchain
    await this.registry.connect(wallet).addCommuter(commuterData);
    console.log(`Commuter ${commuterAgent.unique_id} registered on blockchain at ${wallet.address}`);
    
    return wallet.address;
  }

  /**
   * Register a service provider from the ABM model
   * @param {Object} providerAgent - The provider agent from the ABM model
   * @returns {string} - The account address
   */
  async registerProvider(providerAgent) {
    // Create a new wallet for this provider if not already registered
    if (!this.accounts[providerAgent.unique_id]) {
      const wallet = ethers.Wallet.createRandom().connect(ethers.provider);
      this.accounts[providerAgent.unique_id] = wallet;
      
      // Fund the account with ETH for gas
      await this.admin.sendTransaction({
        to: wallet.address,
        value: ethers.utils.parseEther("0.1")
      });
      
      // Fund with mock tokens
      await this.mockToken.connect(this.admin).transfer(
        wallet.address, 
        ethers.utils.parseEther("1000")
      );
    }
    
    const wallet = this.accounts[providerAgent.unique_id];
    
    // Map mode type to enum
    let modeType = 2; // Default to bus
    if (providerAgent.company_name.includes("Bike")) {
      modeType = 4; // Bike
    } else if (providerAgent.company_name.includes("Uber")) {
      modeType = 3; // Car
    }
    
    // Create provider data object for blockchain
    const providerData = {
      providerId: providerAgent.unique_id,
      providerAddress: wallet.address,
      companyName: providerAgent.company_name,
      modeType: modeType,
      basePrice: ethers.utils.parseEther(providerAgent.base_price.toString()),
      capacity: providerAgent.capacity
    };
    
    // Register on blockchain
    await this.registry.connect(wallet).addProvider(
      providerAgent.unique_id,
      providerData
    );
    
    console.log(`Provider ${providerAgent.unique_id} (${providerAgent.company_name}) registered on blockchain at ${wallet.address}`);
    
    return wallet.address;
  }

  /**
   * Create a travel request on the blockchain
   * @param {Object} commuterAgent - The commuter agent
   * @param {Object} request - The request object from ABM
   * @returns {number} - The blockchain request ID
   */
  async createTravelRequest(commuterAgent, requestObj) {
    const wallet = this.accounts[commuterAgent.unique_id];
    if (!wallet) {
      throw new Error(`Commuter ${commuterAgent.unique_id} not registered`);
    }
    
    // Map purpose to enum
    const purposeMap = {
      'work': 0,
      'school': 1,
      'shopping': 2,
      'medical': 3,
      'trip': 4
    };
    
    // Convert requirements
    const requirements = commuterAgent.get_personal_requirements();
    const requirementKeys = Object.keys(requirements);
    const requirementValues = requirementKeys.map(key => requirements[key]);
    
    // Calculate path info (off-chain)
    const pathInfo = JSON.stringify({
      route: [requestObj.origin, requestObj.destination],
      distance: this.calculateDistance(requestObj.origin, requestObj.destination),
      estimatedTime: 30 // Default estimate
    });
    
    // Create request through facade
    await this.facade.connect(wallet).createTravelRequest(
      requestObj.request_id,
      requestObj.origin,
      requestObj.destination,
      requestObj.start_time,
      purposeMap[requestObj.travel_purpose] || 0,
      requestObj.flexible_time,
      requirementValues,
      requirementKeys,
      pathInfo
    );
    
    console.log(`Travel request ${requestObj.request_id} created on blockchain`);
    return requestObj.request_id;
  }

  /**
   * Submit an offer for a travel request
   * @param {Object} providerAgent - The provider agent
   * @param {number} requestId - The request ID
   * @param {Object} offerDetails - The offer details from ABM
   * @returns {boolean} - Success status
   */
  async submitOffer(providerAgent, requestId, offerDetails) {
    const wallet = this.accounts[providerAgent.unique_id];
    if (!wallet) {
      throw new Error(`Provider ${providerAgent.unique_id} not registered`);
    }
    
    // Map mode to enum
    let mode = 3; // Default to car
    if (providerAgent.company_name.includes("Bike")) {
      mode = 4; // Bike
    }
    
    // Prepare offer data
    const offerData = {
      id: 0, // Will be set by contract
      requestId: requestId,
      providerId: providerAgent.unique_id,
      auctionId: requestId, // Assuming 1:1 mapping
      price: ethers.utils.parseEther(offerDetails.price.toString()),
      mode: mode,
      startTime: offerDetails.start_time,
      totalTime: offerDetails.time * 60, // Convert to seconds
      totalPrice: ethers.utils.parseEther(offerDetails.price.toString()),
      routeDetails: JSON.stringify({
        route: offerDetails.route,
        estimatedTime: offerDetails.time,
        distance: this.calculateDistance(offerDetails.route[0], offerDetails.route[offerDetails.route.length-1])
      })
    };
    
    // Submit offer
    await this.auction.connect(wallet).submitOffer(offerData);
    console.log(`Provider ${providerAgent.unique_id} submitted offer for request ${requestId}`);
    return true;
  }

  /**
   * Finalize an auction to determine winning offer
   * @param {number} requestId - The request ID
   * @returns {Object} - The winning offer details
   */
  async finalizeAuction(requestId) {
    // Finalize the auction (using admin for simplicity)
    await this.auction.connect(this.admin).finalizeAuction(requestId);
    
    // Get winning offers
    const winningOffers = await this.auction.getWinningOffers(requestId);
    
    console.log(`Auction for request ${requestId} finalized`);
    return winningOffers[0]; // Return the first winning offer
  }

  /**
   * Complete service agreement and mint NFT
   * @param {Object} commuterAgent - The commuter agent
   * @param {number} requestId - The request ID
   * @param {number} providerId - The provider ID
   * @param {Object} serviceDetails - The service details
   * @returns {number} - The NFT token ID
   */
  async completeServiceAgreement(commuterAgent, requestId, providerId, serviceDetails) {
    const wallet = this.accounts[commuterAgent.unique_id];
    if (!wallet) {
      throw new Error(`Commuter ${commuterAgent.unique_id} not registered`);
    }
    
    // Approve token spending
    const amount = ethers.utils.parseEther(serviceDetails.price.toString());
    await this.mockToken.connect(wallet).approve(this.nft.address, amount);
    
    // Prepare route details
    const routeDetails = JSON.stringify({
      route: serviceDetails.route,
      distance: this.calculateDistance(serviceDetails.route[0], serviceDetails.route[serviceDetails.route.length-1]),
      estimatedTime: serviceDetails.time,
      completed: true
    });
    
    // Complete service agreement
    const tx = await this.facade.connect(wallet).completeServiceAgreement(
      requestId,
      providerId,
      amount,
      routeDetails,
      serviceDetails.start_time,
      serviceDetails.time * 60, // Convert to seconds
      "ipfs://QmServiceMetadata" // Placeholder URI
    );
    
    // Get token ID from event
    const receipt = await tx.wait();
    const event = receipt.events.find(e => e.event === 'ServiceTokenized');
    const tokenId = event.args.tokenId;
    
    console.log(`Service agreement completed for request ${requestId}, NFT minted with ID ${tokenId}`);
    return tokenId;
  }

  /**
   * Helper method to calculate distance between two points
   * @param {Array} point1 - [x, y] coordinates
   * @param {Array} point2 - [x, y] coordinates
   * @returns {number} - The Euclidean distance
   */
  calculateDistance(point1, point2) {
    return Math.sqrt(
      Math.pow(point2[0] - point1[0], 2) + 
      Math.pow(point2[1] - point1[1], 2)
    );
  }
}

module.exports = BlockchainABMInterface;