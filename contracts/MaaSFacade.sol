// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./MaaSRegistry.sol";
import "./MaaSRequest.sol";
import "./MaaSAuction.sol";
import "./MaaSNFT.sol";
import "./MaaSMarket.sol";

/**
 * @title MaaSFacade
 * @dev Simplifies interaction with the MaaS ecosystem by providing a
 * unified interface for common workflows. Acts as a single entry point
 * for many multi-contract operations.
 */
contract MaaSFacade {
    // ================ State Variables ================
    
    MaaSRegistry public registry;
    MaaSRequest public requestContract;
    MaaSAuction public auctionContract;
    MaaSNFT public nftContract;
    MaaSMarket public marketContract;
    address public admin;
    
    // Statistics tracking
    uint256 public totalTripsCompleted;
    uint256 public totalServicesTokenized;
    mapping(address => uint256) public userActivityScore;
    
    // Integrated reward system
    mapping(address => uint256) public systemPoints;
    mapping(string => uint256) public activityPointValues;
    
    // System performance metrics
    struct SystemMetrics {
        uint256 timestamp;
        uint256 activeUsers;
        uint256 activeRequests;
        uint256 completedTrips;
        uint256 averageResponseTime; // in seconds
        uint256 platformFeeRevenue;
    }
    SystemMetrics[] public historicalMetrics;
    uint256 public lastMetricsUpdate;
    uint256 public metricsUpdateInterval = 86400; // 1 day in seconds
    
    // ================ Events ================
    
    event WorkflowCompleted(
        string workflowType,
        address indexed user,
        uint256 indexed requestId,
        uint256 timestamp
    );
    
    event UserRewarded(
        address indexed user,
        uint256 points,
        string activityType
    );
    
    event MetricsUpdated(
        uint256 timestamp,
        uint256 activeUsers,
        uint256 completedTrips
    );
    
    // ================ Constructor ================
    
    /**
     * @dev Initialize the facade with references to all system contracts
     */
    constructor(
        address _registryAddress,
        address _requestAddress,
        address _auctionAddress,
        address _nftAddress,
        address _marketAddress
    ) {
        registry = MaaSRegistry(_registryAddress);
        requestContract = MaaSRequest(_requestAddress);
        auctionContract = MaaSAuction(_auctionAddress);
        nftContract = MaaSNFT(_nftAddress);
        marketContract = MaaSMarket(_marketAddress);
        admin = msg.sender;
        
        // Set up default point values for different activities
        activityPointValues["create_request"] = 2;
        activityPointValues["complete_service"] = 10;
        activityPointValues["tokenize_service"] = 5;
        activityPointValues["list_service"] = 3;
        activityPointValues["purchase_service"] = 5;
        
        // Initialize metrics
        lastMetricsUpdate = block.timestamp;
    }
    
    // ================ Core Workflow Functions ================
    
    /**
     * @dev Complete workflow: Create travel request and start auction
     * @param requestId Unique ID for the request
     * @param origin Starting coordinates [x,y]
     * @param destination End coordinates [x,y]
     * @param startTime When the trip starts
     * @param travelPurpose Purpose of travel (work, school, etc.)
     * @param flexibleTime Time flexibility description
     * @param requirementValues Boolean flags for requirements
     * @param requirementKeys Requirement type names
     * @param pathInfo Path calculation results (JSON)
     * @return Created request ID
     */
    function createTravelRequest(
        uint256 requestId,
        uint256[] memory origin,
        uint256[] memory destination,
        uint256 startTime,
        MaaSRequest.Purpose travelPurpose,
        string memory flexibleTime,
        bool[] memory requirementValues,
        string[] memory requirementKeys,
        string memory pathInfo
    ) public returns (uint256) {
        // Create request
        requestContract.createRequest(
            requestId,
            origin,
            destination,
            startTime,
            travelPurpose,
            flexibleTime,
            requirementValues,
            requirementKeys,
            msg.sender  // Pass the sender as the owner
        );
        
        // Start auction
        auctionContract.createAuction(requestId, pathInfo);
        
        // Award points to user
        _awardPoints(msg.sender, activityPointValues["create_request"], "create_request");
        
        // Update activity score
        userActivityScore[msg.sender] += 1;
        
        emit WorkflowCompleted("create_request", msg.sender, requestId, block.timestamp);
        
        return requestId;
    }
    
    /**
     * @dev Complete workflow: Pay for service and mint NFT
     * @param requestId Request ID
     * @param providerId Service provider ID
     * @param amount Payment amount
     * @param routeDetails Service route details (JSON)
     * @param startTime Service start time
     * @param duration Service duration
     * @param tokenURI URI for NFT metadata
     * @return NFT token ID
     */
    function completeServiceAgreement(
        uint256 requestId,
        uint256 providerId,
        uint256 amount,
        string memory routeDetails,
        uint256 startTime,
        uint256 duration,
        string memory tokenURI
    ) public returns (uint256) {
        // Pay for service
        nftContract.payForService(requestId, providerId, amount);
        
        // Mint NFT
        uint256 tokenId = nftContract.mintServiceNFT(
            requestId,
            providerId,
            routeDetails,
            amount,
            startTime,
            duration,
            tokenURI
        );
        
        // Update request status
        requestContract.updateRequestStatus(requestId, MaaSRequest.Status.Finished);
        
        // Track statistics
        totalTripsCompleted += 1;
        totalServicesTokenized += 1;
        
        // Award points
        _awardPoints(msg.sender, activityPointValues["complete_service"], "complete_service");
        _awardPoints(msg.sender, activityPointValues["tokenize_service"], "tokenize_service");
        
        // Update activity score
        userActivityScore[msg.sender] += 5;
        
        emit WorkflowCompleted("complete_service", msg.sender, requestId, block.timestamp);
        
        return tokenId;
    }
    
    /**
     * @dev Complete workflow: List service NFT on marketplace
     * @param tokenId NFT token ID
     * @param price Listing price
     * @param useDynamicPricing Whether to use dynamic pricing
     * @param finalPrice Final price if using dynamic pricing
     * @param decayDuration Time for price to decay if using dynamic pricing
     */
    function listServiceForSale(
        uint256 tokenId,
        uint256 price,
        bool useDynamicPricing,
        uint256 finalPrice,
        uint256 decayDuration
    ) public {
        // Verify service still valid
        require(!nftContract.isExpired(tokenId), "Cannot list expired service");
        
        if (useDynamicPricing) {
            marketContract.listNFTWithDynamicPricing(
                tokenId,
                price,
                finalPrice,
                decayDuration
            );
        } else {
            marketContract.listNFTForSale(tokenId, price);
        }
        
        // Award points
        _awardPoints(msg.sender, activityPointValues["list_service"], "list_service");
        
        // Update activity score
        userActivityScore[msg.sender] += 2;
        
        emit WorkflowCompleted("list_service", msg.sender, tokenId, block.timestamp);
    }
    
    /**
     * @dev Complete workflow: Find, purchase, and approve a service in marketplace
     * @param tokenId NFT token ID to purchase
     * @return Success status
     */
    function findAndPurchaseService(uint256 tokenId) public returns (bool) {
        // Purchase NFT
        marketContract.purchaseNFT(tokenId);
        
        // Award points
        _awardPoints(msg.sender, activityPointValues["purchase_service"], "purchase_service");
        
        // Update activity score
        userActivityScore[msg.sender] += 3;
        
        emit WorkflowCompleted("purchase_service", msg.sender, tokenId, block.timestamp);
        
        return true;
    }
    
    /**
     * @dev Complete workflow: Create service bundle from multiple services
     * @param tokenIds Array of NFT token IDs
     * @param name Bundle name
     * @param price Bundle price
     * @return Bundle ID
     */
    function createAndListServiceBundle(
        uint256[] memory tokenIds,
        string memory name,
        uint256 price
    ) public returns (uint256) {
        // Create bundle
        uint256 bundleId = nftContract.createServiceBundle(
            tokenIds,
            name,
            price
        );
        
        // Award points (bundle creation already awards points in NFT contract)
        
        // Update activity score
        userActivityScore[msg.sender] += 4;
        
        emit WorkflowCompleted("create_bundle", msg.sender, bundleId, block.timestamp);
        
        return bundleId;
    }
    
    // ================ Helper Functions ================
    
    /**
     * @dev Get active auctions for searching available services
     * @return Array of active auction IDs
     */
    function getActiveAuctions() public view returns (uint256[] memory) {
        // Implementation depends on how you want to track active auctions
        // This is a simplified example that delegates to the auction contract
        return auctionContract.getActiveAuctions();
    }
    
    /**
     * @dev Search for services matching user needs
     * @param minPrice Minimum price
     * @param maxPrice Maximum price
     * @param minDeparture Minimum departure time
     * @param maxDeparture Maximum departure time
     * @param origin Origin area coordinates [x, y, radius]
     * @param destination Destination area coordinates [x, y, radius]
     * @return Array of matching services
     */
    function searchAvailableServices(
        uint256 minPrice,
        uint256 maxPrice,
        uint256 minDeparture,
        uint256 maxDeparture,
        uint256[] memory origin,
        uint256[] memory destination
    ) public view returns (MaaSMarket.SellRequest[] memory) {
        // Create search parameters
        MaaSMarket.SearchParams memory params = MaaSMarket.SearchParams({
            minPrice: minPrice,
            maxPrice: maxPrice,
            minDeparture: minDeparture,
            maxDeparture: maxDeparture,
            originArea: origin,
            destArea: destination,
            provider: address(0), // No specific provider filter
            serviceType: 0, // No specific service type filter
            includeExpired: false // Exclude expired services
        });
        
        return marketContract.advancedSearch(params);
    }
    
    /**
     * @dev Calculate and recommend services based on user history and preferences
     * @param commuterId User's commuter ID
     * @return Array of recommended service IDs
     */
    function getRecommendedServices(uint256 commuterId) public view returns (uint256[] memory) {
        // This would require a more complex implementation
        // For now, return a simplified version
        
        // Get commuter's address
        address commuterAddress = registry.commuterIdToAddress(commuterId);
        
        // Get basic commuter data
        (uint256 userCommuterId, , , , , ) = registry.getCommuter(commuterAddress);
        
        // Get active listings
        MaaSMarket.SellRequest[] memory listings = marketContract.getActiveListings();
        
        // This is a placeholder - in a real implementation, you would:
        // 1. Analyze commuter's past trips from request history
        // 2. Look at commuter's preferences and requirements
        // 3. Find matching services in active listings
        // 4. Sort by relevance
        
        // For now, return first 5 listings if available
        uint256[] memory recommendations = new uint256[](listings.length > 5 ? 5 : listings.length);
        for (uint256 i = 0; i < recommendations.length; i++) {
            recommendations[i] = listings[i].tokenId;
        }
        
        return recommendations;
    }
    
    // ================ Reward System Functions ================
    
    /**
     * @dev Award points to a user for an activity
     * @param user User address
     * @param points Points to award
     * @param activityType Activity description
     */
    function _awardPoints(address user, uint256 points, string memory activityType) internal {
        systemPoints[user] += points;
        emit UserRewarded(user, points, activityType);
    }
    
    /**
     * @dev Admin can adjust point values for different activities
     * @param activityType Activity type
     * @param points New point value
     */
    function setActivityPointValue(string memory activityType, uint256 points) public {
        require(msg.sender == admin, "Only admin can set point values");
        activityPointValues[activityType] = points;
    }
    
    /**
     * @dev Award bonus points to a user
     * @param user User address
     * @param points Bonus points
     * @param reason Reason for bonus
     */
    function awardBonusPoints(address user, uint256 points, string memory reason) public {
        require(msg.sender == admin, "Only admin can award bonus points");
        systemPoints[user] += points;
        emit UserRewarded(user, points, reason);
    }
    
    /**
     * @dev Get a user's total points
     * @param user User address
     * @return Point balance
     */
    function getUserPoints(address user) public view returns (uint256) {
        return systemPoints[user];
    }
    
    // ================ System Metrics Functions ================
    
    /**
     * @dev Update system metrics for analytics
     */
    function updateSystemMetrics() public {
        // Only update if interval has passed
        if (block.timestamp < lastMetricsUpdate + metricsUpdateInterval) {
            return;
        }
        
        // Calculate metrics
        uint256 activeUsers = _countActiveUsers();
        uint256 activeRequests = requestContract.getActiveRequestCount();
        uint256 newCompletedTrips = totalTripsCompleted;
        uint256 responseTime = _calculateAverageResponseTime();
        uint256 feeRevenue = _calculatePlatformFeeRevenue();
        
        // Create metrics entry
        SystemMetrics memory newMetrics = SystemMetrics({
            timestamp: block.timestamp,
            activeUsers: activeUsers,
            activeRequests: activeRequests,
            completedTrips: newCompletedTrips,
            averageResponseTime: responseTime,
            platformFeeRevenue: feeRevenue
        });
        
        // Store metrics
        historicalMetrics.push(newMetrics);
        lastMetricsUpdate = block.timestamp;
        
        emit MetricsUpdated(block.timestamp, activeUsers, newCompletedTrips);
    }
    
    /**
     * @dev Get latest system metrics
     * @return Latest metrics
     */
    function getLatestMetrics() public view returns (SystemMetrics memory) {
        if (historicalMetrics.length == 0) {
            return SystemMetrics(0, 0, 0, 0, 0, 0);
        }
        return historicalMetrics[historicalMetrics.length - 1];
    }
    
    /**
     * @dev Count active users in recent period
     * @return Count of active users
     */
    function _countActiveUsers() internal view returns (uint256) {
        // This would require analyzing transaction history
        // For now, return a simple count of users with activity score > 0
        uint256 count = 0;
        uint256 commuterCount = registry.numberOfCommuters();
        address[] memory commuters = new address[](commuterCount);
        for (uint256 i = 1; i <= commuterCount; i++) {
            address commuterAddr = registry.commuterIdToAddress(i);
            if (commuterAddr != address(0)) {
                commuters[i-1] = commuterAddr;
            }
        }
        
        for (uint256 i = 0; i < commuters.length; i++) {
            if (userActivityScore[commuters[i]] > 0) {
                count++;
            }
        }
        
        return count;
    }
    
    /**
     * @dev Calculate average response time for auctions
     * @return Average response time in seconds
     */
    function _calculateAverageResponseTime() internal view returns (uint256) {
        // This would require analyzing auction data
        // For simplicity, return a placeholder value
        return 300; // 5 minutes
    }
    
    /**
     * @dev Calculate total platform fee revenue
     * @return Total fee revenue
     */
    function _calculatePlatformFeeRevenue() internal view returns (uint256) {
        // This would require analyzing market transactions
        // For simplicity, return a placeholder value
        return 1000;
    }
    
    /**
     * @dev Set metrics update interval
     * @param newInterval New interval in seconds
     */
    function setMetricsUpdateInterval(uint256 newInterval) public {
        require(msg.sender == admin, "Only admin can set update interval");
        require(newInterval >= 3600, "Interval must be at least 1 hour");
        metricsUpdateInterval = newInterval;
    }
}