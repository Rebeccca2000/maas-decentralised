// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./MaaSRegistry.sol";
import "./MaaSRequest.sol";

/**
 * @title MaaSAuction
 * @dev Auction mechanism for matching mobility service requests with providers
 * Enhanced with multiple auction types, off-chain calculation support,
 * and incentive mechanisms
 */
contract MaaSAuction {
    // ================ Enumerations ================
    
    enum AuctionType { Standard, FixedPrice, Dutch, EnglishAscending, SmartMatching }
    enum AuctionStatus { Active, Finalized, Cancelled, Expired }
    
    // ================ Data Structures ================
    
    /**
     * @dev Service offer from a provider
     */
    struct Offer {
        uint256 id;                     // Offer ID
        uint256 requestId;              // Associated request ID
        uint256 providerId;             // Provider ID
        uint256 auctionId;              // Auction ID
        uint256 price;                  // Offered price
        MaaSRegistry.TransportMode mode;// Transport mode
        uint256 startTime;              // Service start time
        uint256 totalTime;              // Total service time
        uint256 totalPrice;             // Total price including fees
        string routeDetails;            // JSON with route details
        
        // Enhanced offer attributes
        uint256 offerTime;              // When offer was submitted
        uint256 capacity;               // Available capacity for this service
        uint256 quality;                // Provider's quality score
        uint256 reliability;            // Provider's reliability score
        bool isVerified;                // Provider verification status
        bool hasInsurance;              // Whether offer includes insurance
        string[] additionalServices;    // List of additional services
        bool isConditional;             // Whether offer is conditional
        string conditions;              // Conditions for the offer
    }
    
    /**
     * @dev Auction for a mobility service
     */
    struct Auction {
        uint256 auctionId;              // Auction ID
        string pathInfo;                // Path calculated off-chain
        AuctionStatus status;           // Auction status
        uint256 requestId;              // Associated request ID
        uint256 commuterId;             // Commuter ID
        uint256[] winningOfferIds;      // Winning offer IDs
        uint256 startTime;              // Auction start time
        uint256 endTime;                // Auction end time
        
        // Enhanced auction attributes
        AuctionType auctionType;        // Type of auction
        uint256 reservePrice;           // Minimum acceptable price
        uint256 targetPrice;            // Target price (for smart matching)
        bytes calculationProof;         // Proof of off-chain calculation
        uint256 rewardAmount;           // Reward for participation
        uint256 winnerDeterminationTime;// When winner was determined
        uint256 offerCount;             // Number of offers received
        bool requiresVerified;          // Only verified providers
        uint256 minProviderRating;      // Minimum provider rating
        mapping(uint256 => uint256) bidderActivity; // Track activity by provider
    }
    
    /**
     * @dev Smart matching parameters
     */
    struct MatchingParameters {
        uint256 maxPrice;               // Maximum price willing to pay
        uint256 preferredStartTime;     // Preferred start time
        uint256 timeFlexibility;        // How flexible on time (+/- minutes)
        uint256 qualityWeight;          // 0-100 weight for provider quality
        uint256 priceWeight;            // 0-100 weight for price
        uint256 reliabilityWeight;      // 0-100 weight for reliability
        uint256 speedWeight;            // 0-100 weight for speed/time
        MaaSRegistry.TransportMode preferredMode; // Preferred transport mode
        bool onlyVerifiedProviders;     // Require verified providers
        uint256 originRadius;           // Distance flexibility from origin
        uint256 destinationRadius;      // Distance flexibility to destination
    }
    
    /**
     * @dev Service completion confirmation
     */
    struct Confirmation {
        uint256 requestId;              // Request ID
        uint256 auctionId;              // Auction ID
        uint256 offerId;                // Offer ID
        address provider;               // Provider address
        address commuter;               // Commuter address
        uint256 timestamp;              // Confirmation time
        bool completed;                 // Whether service was completed
        string feedback;                // Service feedback
        uint256 rating;                 // Service rating (0-100)
    }
    
    // ================ State Variables ================
    
    // Core auction data
    mapping(uint256 => Auction) public auctions;
    mapping(uint256 => Offer[]) public offersByAuction;
    mapping(uint256 => mapping(uint256 => bool)) public hasProviderSubmitted;  // auctionId => providerId => hasSubmitted
    
    // Enhanced tracking
    mapping(uint256 => MatchingParameters) public matchingParameters;
    mapping(uint256 => mapping(uint256 => Confirmation)) public serviceConfirmations; // auctionId => offerId => confirmation
    mapping(address => uint256) public commuterBonusPoints;
    mapping(address => uint256) public providerBonusPoints;
    
    // Contract references and counters
    uint256 public numberOfAuctions = 0;
    MaaSRegistry public registry;
    MaaSRequest public requestContract;
    address public admin;
    
    // Platform parameters
    uint256 public minAuctionDuration = 5 minutes;
    uint256 public maxAuctionDuration = 24 hours;
    uint256 public defaultAuctionDuration = 1 hours;
    uint256 public platformFeePercentage = 2; // 2% platform fee
    bool public feesEnabled = true;
    bool public emergencyPause = false;
    
    // ================ Events ================
    
    event AuctionCreated(uint256 auctionId, uint256 requestId, uint256 commuterId, AuctionType auctionType);
    event OfferSubmitted(uint256 auctionId, uint256 offerId, uint256 providerId, uint256 price);
    event AuctionFinalized(uint256 auctionId, uint256[] winningOfferIds);
    event AuctionCancelled(uint256 auctionId, string reason);
    event ServiceConfirmed(uint256 auctionId, uint256 offerId, bool successful, uint256 rating);
    event RewardIssued(address recipient, uint256 amount, string reason);
    event SmartMatchingCompleted(uint256 auctionId, uint256 bestMatchOfferId);
    
    // ================ Modifiers ================
    
    modifier notPaused() {
        require(!emergencyPause, "Platform is currently paused");
        _;
    }
    
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can call this function");
        _;
    }
    
    modifier onlyCommuter(uint256 commuterId) {
        address commuterAddress = registry.commuterIdToAddress(commuterId);
        require(msg.sender == commuterAddress, "Only the commuter can call this function");
        _;
    }
    
    modifier onlyProvider(uint256 providerId) {
        address providerAddress = registry.providerIdToAddress(providerId);
        require(msg.sender == providerAddress, "Only the provider can call this function");
        _;
    }
    
    modifier auctionExists(uint256 auctionId) {
        require(auctions[auctionId].auctionId == auctionId, "Auction does not exist");
        _;
    }
    
    // ================ Constructor ================
    
    /**
     * @dev Initialize the auction contract with contract references
     * @param _registryAddress Address of the registry contract
     * @param _requestAddress Address of the request contract
     */
    constructor(address _registryAddress, address _requestAddress) {
        registry = MaaSRegistry(_registryAddress);
        requestContract = MaaSRequest(_requestAddress);
        admin = msg.sender;
    }
    
    // ================ Auction Creation Functions ================
    
    /**
     * @dev Create a standard auction for a travel request
     * @param requestId Request ID
     * @param pathInfo Path information (JSON)
     * @return Auction ID
     */
    function createAuction(uint256 requestId, string memory pathInfo) public notPaused returns (uint256) {
        // Get request basic info
        (uint256 _requestId, uint256 commuterId, , , , , MaaSRequest.Status status, ) = 
            requestContract.getRequestBasicInfo(requestId);
        
        // Check if the request exists and is active
        require(_requestId == requestId, "Request does not exist");
        require(status == MaaSRequest.Status.Active, "Request is not active");
        
        // Create new auction
        numberOfAuctions++;
        uint256 auctionId = numberOfAuctions;
        
        uint256[] memory emptyArray = new uint256[](0);
        
        // Setup auction basic attributes in storage
        Auction storage newAuction = auctions[auctionId];
        newAuction.auctionId = auctionId;
        newAuction.pathInfo = pathInfo;
        newAuction.status = AuctionStatus.Active;
        newAuction.requestId = requestId;
        newAuction.commuterId = commuterId;
        newAuction.winningOfferIds = emptyArray;
        newAuction.startTime = block.timestamp;
        newAuction.endTime = block.timestamp + defaultAuctionDuration;
        
        // Setup enhanced attributes
        newAuction.auctionType = AuctionType.Standard;
        newAuction.reservePrice = 0; // No reserve price by default
        newAuction.rewardAmount = calculateRewardAmount(commuterId);
        newAuction.offerCount = 0;
        newAuction.requiresVerified = false;
        newAuction.minProviderRating = 0;
        
        // Award participation points
        address commuterAddress = registry.commuterIdToAddress(commuterId);
        if (commuterAddress != address(0)) {
            awardCommuterPoints(commuterAddress, 5, "auction_creation");
        }
        
        emit AuctionCreated(auctionId, requestId, commuterId, AuctionType.Standard);
        
        return auctionId;
    }
    
    /**
     * @dev Create an auction with specific type and parameters
     * @param requestId Request ID
     * @param pathInfo Path information (JSON)
     * @param auctionType Type of auction
     * @param duration Auction duration in seconds
     * @param reservePrice Minimum acceptable price
     * @param requiresVerified Whether only verified providers can participate
     * @param minProviderRating Minimum provider rating requirement
     * @return Auction ID
     */
    function createAdvancedAuction(
        uint256 requestId, 
        string memory pathInfo,
        AuctionType auctionType,
        uint256 duration,
        uint256 reservePrice,
        bool requiresVerified,
        uint256 minProviderRating
    ) public notPaused returns (uint256) {
        // Get request basic info
        (uint256 _requestId, uint256 commuterId, , , , , MaaSRequest.Status status, ) = 
            requestContract.getRequestBasicInfo(requestId);
        
        // Check if the request exists and is active
        require(_requestId == requestId, "Request does not exist");
        require(status == MaaSRequest.Status.Active, "Request is not active");
        
        // Validate parameters
        require(duration >= minAuctionDuration && duration <= maxAuctionDuration, 
                "Duration outside allowed range");
        require(minProviderRating <= 100, "Rating must be 0-100");
        
        // Create new auction
        numberOfAuctions++;
        uint256 auctionId = numberOfAuctions;
        
        uint256[] memory emptyArray = new uint256[](0);
        
        // Setup auction in storage
        Auction storage newAuction = auctions[auctionId];
        newAuction.auctionId = auctionId;
        newAuction.pathInfo = pathInfo;
        newAuction.status = AuctionStatus.Active;
        newAuction.requestId = requestId;
        newAuction.commuterId = commuterId;
        newAuction.winningOfferIds = emptyArray;
        newAuction.startTime = block.timestamp;
        newAuction.endTime = block.timestamp + duration;
        
        // Setup enhanced attributes
        newAuction.auctionType = auctionType;
        newAuction.reservePrice = reservePrice;
        newAuction.rewardAmount = calculateRewardAmount(commuterId);
        newAuction.offerCount = 0;
        newAuction.requiresVerified = requiresVerified;
        newAuction.minProviderRating = minProviderRating;
        
        // Award participation points with bonus for advanced auction
        address commuterAddress = registry.commuterIdToAddress(commuterId);
        if (commuterAddress != address(0)) {
            awardCommuterPoints(commuterAddress, 10, "advanced_auction_creation");
        }
        
        emit AuctionCreated(auctionId, requestId, commuterId, auctionType);
        
        return auctionId;
    }
    
    /**
     * @dev Create a smart matching auction with detailed parameters
     * @param requestId Request ID
     * @param pathInfo Path information (JSON)
     * @param params Smart matching parameters
     * @return Auction ID
     */
    function createSmartMatchingAuction(
        uint256 requestId,
        string memory pathInfo,
        MatchingParameters memory params
    ) public notPaused returns (uint256) {
        // Get request basic info
        (uint256 _requestId, uint256 commuterId, , , , , MaaSRequest.Status status, ) = 
            requestContract.getRequestBasicInfo(requestId);
        
        // Check if the request exists and is active
        require(_requestId == requestId, "Request does not exist");
        require(status == MaaSRequest.Status.Active, "Request is not active");
        
        // Validate weights
        uint256 totalWeight = params.qualityWeight + params.priceWeight + 
                             params.reliabilityWeight + params.speedWeight;
        require(totalWeight == 100, "Weights must sum to 100");
        
        // Create new auction
        numberOfAuctions++;
        uint256 auctionId = numberOfAuctions;
        
        uint256[] memory emptyArray = new uint256[](0);
        
        // Setup auction
        Auction storage newAuction = auctions[auctionId];
        newAuction.auctionId = auctionId;
        newAuction.pathInfo = pathInfo;
        newAuction.status = AuctionStatus.Active;
        newAuction.requestId = requestId;
        newAuction.commuterId = commuterId;
        newAuction.winningOfferIds = emptyArray;
        newAuction.startTime = block.timestamp;
        newAuction.endTime = block.timestamp + defaultAuctionDuration;
        
        // Setup enhanced attributes
        newAuction.auctionType = AuctionType.SmartMatching;
        newAuction.reservePrice = params.maxPrice;
        newAuction.targetPrice = params.maxPrice / 2; // Target 50% of max as optimal
        newAuction.rewardAmount = calculateRewardAmount(commuterId);
        newAuction.offerCount = 0;
        newAuction.requiresVerified = params.onlyVerifiedProviders;
        newAuction.minProviderRating = 0;
        
        // Store matching parameters
        matchingParameters[auctionId] = params;
        
        // Award participation points with bonus for smart matching
        address commuterAddress = registry.commuterIdToAddress(commuterId);
        if (commuterAddress != address(0)) {
            awardCommuterPoints(commuterAddress, 15, "smart_matching_creation");
        }
        
        emit AuctionCreated(auctionId, requestId, commuterId, AuctionType.SmartMatching);
        
        return auctionId;
    }
    
    /**
     * @dev Get active auction IDs
     * @return Array of active auction IDs
     */
    function getActiveAuctions() public view returns (uint256[] memory) {
        uint256 activeCount = 0;
        for (uint256 i = 1; i <= numberOfAuctions; i++) {
            if (auctions[i].status == AuctionStatus.Active) {
                activeCount++;
            }
        }
        
        uint256[] memory activeAuctions = new uint256[](activeCount);
        uint256 currentIndex = 0;
        
        for (uint256 i = 1; i <= numberOfAuctions; i++) {
            if (auctions[i].status == AuctionStatus.Active) {
                activeAuctions[currentIndex] = i;
                currentIndex++;
            }
        }
        
        return activeAuctions;
    }
    // ================ Offer Submission Functions ================
    
    /**
     * @dev Submit an offer for an auction
     * @param newOffer Offer details
     * @return Offer ID
     */
    function submitOffer(Offer memory newOffer) public notPaused returns (uint256) {
        // Verify the auction exists and is active
        Auction storage auction = auctions[newOffer.auctionId];
        require(auction.status == AuctionStatus.Active, "Auction is not active");
        require(block.timestamp < auction.endTime, "Auction has ended");
        
        // Verify the provider is registered
        address providerAddress = msg.sender;
        MaaSRegistry.Provider memory provider = registry.getProvider(providerAddress);
        require(provider.providerId > 0, "Provider is not registered");
        require(provider.isActive, "Provider is not active");
        
        // Check if this provider has already submitted an offer for this auction
        require(!hasProviderSubmitted[newOffer.auctionId][provider.providerId], 
                "Provider has already submitted an offer for this auction");
        
        // Check if provider meets verification requirements
        if (auction.requiresVerified) {
            bool isVerified = registry.isProviderVerified(providerAddress);
            require(isVerified, "Only verified providers can participate");
            
            // Check rating requirement
            if (auction.minProviderRating > 0) {
                uint256 providerReputation = registry.getProviderReputation(providerAddress);
                require(providerReputation >= auction.minProviderRating, 
                    "Provider rating below minimum requirement");
            }
        }
        
        // Set offer ID and mark provider as having submitted
        uint256 offerId = offersByAuction[newOffer.auctionId].length;
        hasProviderSubmitted[newOffer.auctionId][provider.providerId] = true;
        
        // Enhance offer with available provider data
        Offer memory enhancedOffer = newOffer;
        enhancedOffer.id = offerId;
        enhancedOffer.offerTime = block.timestamp;
        enhancedOffer.capacity = provider.capacity; // Use total capacity instead of availableCapacity
        enhancedOffer.quality = registry.getProviderReputation(providerAddress); // Use reputation as quality
        enhancedOffer.reliability = registry.getProviderReputation(providerAddress); // Use reputation as reliability
        enhancedOffer.isVerified = registry.isProviderVerified(providerAddress);
        
        // Handle auction type specifics
        if (auction.auctionType == AuctionType.FixedPrice) {
            // For fixed price auctions, price must match reserve price
            require(enhancedOffer.price == auction.reservePrice, "Price must match fixed price");
        } else if (auction.auctionType == AuctionType.Dutch) {
            // For Dutch auctions, first acceptable offer wins
            if (enhancedOffer.price >= auction.reservePrice) {
                // Auto-finalize with this offer
                uint256[] memory winningIds = new uint256[](1);
                winningIds[0] = offerId;
                auction.winningOfferIds = winningIds;
                auction.status = AuctionStatus.Finalized;
                auction.winnerDeterminationTime = block.timestamp;
                
                // Add offer to collection
                offersByAuction[newOffer.auctionId].push(enhancedOffer);
                auction.offerCount++;
                
                // Update bidder activity
                auction.bidderActivity[provider.providerId] = block.timestamp;
                
                // Award points for winning Dutch auction
                awardProviderPoints(providerAddress, 20, "dutch_auction_win");
                
                emit OfferSubmitted(newOffer.auctionId, offerId, provider.providerId, enhancedOffer.price);
                emit AuctionFinalized(newOffer.auctionId, winningIds);
                
                return offerId;
            }
        }
        
        // For other auction types, just add the offer
        offersByAuction[newOffer.auctionId].push(enhancedOffer);
        auction.offerCount++;
        
        // Update bidder activity
        auction.bidderActivity[provider.providerId] = block.timestamp;
        
        // Award points for offer submission
        awardProviderPoints(providerAddress, 5, "offer_submission");
        
        // Handle English ascending auction auto-extension
        if (auction.auctionType == AuctionType.EnglishAscending) {
            // If offer comes in last 5 minutes, extend auction by 5 minutes
            if (block.timestamp > auction.endTime - 5 minutes) {
                auction.endTime = block.timestamp + 5 minutes;
            }
        }
        
        // Auto-finalize smart matching if this is a good match
        if (auction.auctionType == AuctionType.SmartMatching) {
            trySmartMatching(newOffer.auctionId);
        }
        
        emit OfferSubmitted(newOffer.auctionId, offerId, provider.providerId, enhancedOffer.price);
        
        return offerId;
    }
    
    // ================ Auction Finalization Functions ================
    
    /**
     * @dev Finalize an auction using simple price-based selection
     * @param auctionId Auction ID
     * @return Success status
     */
    function finalizeAuction(uint256 auctionId) public auctionExists(auctionId) notPaused returns (bool) {
        Auction storage auction = auctions[auctionId];
        
        // Verify the auction exists and is active
        require(auction.status == AuctionStatus.Active, "Auction is not active");
        
        // Anyone can finalize after end time
        // Only admin or commuter can finalize before end time
        if (block.timestamp < auction.endTime) {
            address senderCommuterAddress = registry.commuterIdToAddress(auction.commuterId);
            require(msg.sender == senderCommuterAddress || msg.sender == admin, 
                    "Only commuter or admin can finalize early");
        }
        
        // Get all offers for this auction
        Offer[] storage offers = offersByAuction[auctionId];
        require(offers.length > 0, "No offers to finalize");
        
        uint256 bestOfferId = 0;
        uint256 bestOfferPrice = type(uint256).max;
        
        // Simple algorithm: select lowest price offer meeting reserve price
        for (uint256 i = 0; i < offers.length; i++) {
            // Check reserve price
            if (offers[i].price < auction.reservePrice) {
                continue;
            }
            
            if (offers[i].price < bestOfferPrice) {
                bestOfferPrice = offers[i].price;
                bestOfferId = offers[i].id;
            }
        }
        
        // Verify we found an acceptable offer
        require(bestOfferPrice != type(uint256).max, "No offers meet reserve price");
        
        // Store the winning offer
        uint256[] memory winningOfferIds = new uint256[](1);
        winningOfferIds[0] = bestOfferId;
        auction.winningOfferIds = winningOfferIds;
        auction.status = AuctionStatus.Finalized;
        auction.winnerDeterminationTime = block.timestamp;
        
        // Award finalization bonuses
        address providerAddress = registry.providerIdToAddress(offers[bestOfferId].providerId);
        if (providerAddress != address(0)) {
            awardProviderPoints(providerAddress, 15, "auction_win");
        }
        
        address commuterAddress = registry.commuterIdToAddress(auction.commuterId);
        if (commuterAddress != address(0)) {
            awardCommuterPoints(commuterAddress, 10, "auction_complete");
        }
        
        emit AuctionFinalized(auctionId, winningOfferIds);
        return true;
    }
    
    /**
     * @dev Finalize auction with off-chain calculation results
     * @param auctionId Auction ID
     * @param selectedOfferIds IDs of selected offers
     * @param calculationProof Proof/details of off-chain calculation
     * @return Success status
     */
    function finalizeAuctionWithResults(
        uint256 auctionId, 
        uint256[] memory selectedOfferIds,
        bytes memory calculationProof
    ) public auctionExists(auctionId) notPaused returns (bool) {
        Auction storage auction = auctions[auctionId];
        
        // Verify the auction is active
        require(auction.status == AuctionStatus.Active, "Auction is not active");
        
        // Only the commuter or admin can finalize with specific results
        address commuterAddress = registry.commuterIdToAddress(auction.commuterId);
        require(
            msg.sender == commuterAddress || msg.sender == admin,
            "Not authorized to finalize auction"
        );
        
        // Verify offers exist
        for (uint256 i = 0; i < selectedOfferIds.length; i++) {
            require(
                selectedOfferIds[i] < offersByAuction[auctionId].length,
                "Selected offer does not exist"
            );
        }
        
        // Store the calculation details
        auction.calculationProof = calculationProof;
        
        // Store the winning offers
        auction.winningOfferIds = selectedOfferIds;
        auction.status = AuctionStatus.Finalized;
        auction.winnerDeterminationTime = block.timestamp;
        
        // Award finalization bonuses to all winning providers
        for (uint256 i = 0; i < selectedOfferIds.length; i++) {
            uint256 providerId = offersByAuction[auctionId][selectedOfferIds[i]].providerId;
            address providerAddress = registry.providerIdToAddress(providerId);
            
            if (providerAddress != address(0)) {
                uint256 bonusPoints = 15;
                // Extra bonus for smart matching wins
                if (auction.auctionType == AuctionType.SmartMatching) {
                    bonusPoints += 10;
                }
                awardProviderPoints(providerAddress, bonusPoints, "auction_win");
            }
        }
        
        // Award commuter points
        if (commuterAddress != address(0)) {
            awardCommuterPoints(commuterAddress, 10, "auction_complete");
        }
        
        emit AuctionFinalized(auctionId, selectedOfferIds);
        return true;
    }
    
    /**
     * @dev Cancel an auction
     * @param auctionId Auction ID
     * @param reason Cancellation reason
     * @return Success status
     */
    function cancelAuction(uint256 auctionId, string memory reason) public auctionExists(auctionId) returns (bool) {
        Auction storage auction = auctions[auctionId];
        
        // Verify auction is active
        require(auction.status == AuctionStatus.Active, "Auction is not active");
        
        // Only commuter or admin can cancel
        address commuterAddr = registry.commuterIdToAddress(auction.commuterId);
        require(
            msg.sender == commuterAddr|| msg.sender == admin,
            "Not authorized to cancel auction"
        );
        
        // Update auction status
        auction.status = AuctionStatus.Cancelled;
        
        emit AuctionCancelled(auctionId, reason);
        return true;
    }
    
    // ================ Service Confirmation Functions ================
    
    /**
     * @dev Confirm service completion (by commuter)
     * @param auctionId Auction ID
     * @param offerId Offer ID
     * @param completed Whether service was completed successfully
     * @param rating Service rating (0-100)
     * @param feedback Service feedback
     * @return Success status
     */
    function confirmService(
        uint256 auctionId,
        uint256 offerId,
        bool completed,
        uint256 rating,
        string memory feedback
    ) public auctionExists(auctionId) notPaused returns (bool) {
        Auction storage auction = auctions[auctionId];
        
        // Only commuter can confirm
        address commuterAddress = registry.commuterIdToAddress(auction.commuterId);
        require(msg.sender == commuterAddress, "Only commuter can confirm service");
        
        // Verify auction is finalized
        require(auction.status == AuctionStatus.Finalized, "Auction is not finalized");
        
        // Verify this is a winning offer
        bool isWinningOffer = false;
        for (uint256 i = 0; i < auction.winningOfferIds.length; i++) {
            if (auction.winningOfferIds[i] == offerId) {
                isWinningOffer = true;
                break;
            }
        }
        require(isWinningOffer, "Not a winning offer");
        
        // Check rating range
        require(rating <= 100, "Rating must be 0-100");
        
        // Get the offer provider
        uint256 providerId = offersByAuction[auctionId][offerId].providerId;
        address providerAddress = registry.providerIdToAddress(providerId);
        
        // Create confirmation
        serviceConfirmations[auctionId][offerId] = Confirmation({
            requestId: auction.requestId,
            auctionId: auctionId,
            offerId: offerId,
            provider: providerAddress,
            commuter: commuterAddress,
            timestamp: block.timestamp,
            completed: completed,
            feedback: feedback,
            rating: rating
        });
        
        // Award completion points
        if (completed) {

            // Record trip in registry
            if (isWinningOffer) {
                registry.recordTrip(
                    commuterAddress, 
                    offersByAuction[auctionId][offerId].price
                );
                
                registry.recordServiceActivity(
                    providerAddress
                );
            }
            
            // Award bonus points for good service
            if (rating >= 80) {
                awardProviderPoints(providerAddress, 20, "high_rating_service");
            } else if (rating >= 60) {
                awardProviderPoints(providerAddress, 10, "good_rating_service");
            }
            
            // Commuter gets points for rating
            awardCommuterPoints(commuterAddress, 5, "service_rating");
        }
        
        emit ServiceConfirmed(auctionId, offerId, completed, rating);
        return true;
    }
    
    // ================ Smart Matching Functions ================
    
    /**
     * @dev Try to find a smart match that meets the parameters
     * @param auctionId Auction ID
     * @return Whether a match was found and finalized
     */
    function trySmartMatching(uint256 auctionId) internal returns (bool) {
        Auction storage auction = auctions[auctionId];
        
        // Only applicable for smart matching auctions
        if (auction.auctionType != AuctionType.SmartMatching) {
            return false;
        }
        
        // Get matching parameters
        MatchingParameters storage params = matchingParameters[auctionId];
        
        // Need at least 3 offers for smart matching to be effective
        if (auction.offerCount < 3) {
            return false;
        }
        
        // Calculate scores for each offer
        Offer[] storage offers = offersByAuction[auctionId];
        uint256 bestScore = 0;
        uint256 bestOfferId = 0;
        bool foundGoodMatch = false;
        
        for (uint256 i = 0; i < offers.length; i++) {
            // Skip offers that don't meet basic requirements
            if (offers[i].price > params.maxPrice) {
                continue;
            }
            
            if (params.onlyVerifiedProviders && !offers[i].isVerified) {
                continue;
            }
            
            if (offers[i].mode != params.preferredMode && params.preferredMode != MaaSRegistry.TransportMode.BackgroundTraffic) {
                continue;
            }
            
            // Calculate weighted score (higher is better)
            uint256 score = 0;
            
            // Price score (lower price = higher score)
            uint256 priceScore = params.maxPrice > 0 ? 
                                (params.maxPrice - offers[i].price) * 100 / params.maxPrice : 100;
            score += priceScore * params.priceWeight / 100;
            
            // Quality score
            score += offers[i].quality * params.qualityWeight / 100;
            
            // Reliability score
            score += offers[i].reliability * params.reliabilityWeight / 100;
            
            // Time score (based on proximity to preferred start time)
            uint256 timeDistance = offers[i].startTime > params.preferredStartTime ? 
                                offers[i].startTime - params.preferredStartTime : 
                                params.preferredStartTime - offers[i].startTime;
                                
            uint256 timeScore = 100;
            if (timeDistance > params.timeFlexibility) {
                timeScore = 0;
            } else if (timeDistance > 0) {
                timeScore = 100 - (timeDistance * 100 / params.timeFlexibility);
            }
            
            score += timeScore * params.speedWeight / 100;
            
            // Check if this is the best offer so far
            if (score > bestScore) {
                bestScore = score;
                bestOfferId = i;
                foundGoodMatch = true;
            }
        }
        
        // If we found a good match with a high score, auto-finalize
        // (80% threshold means the offer is very good on most dimensions)
        if (foundGoodMatch && bestScore >= 80) {
            uint256[] memory winningIds = new uint256[](1);
            winningIds[0] = bestOfferId;
            auction.winningOfferIds = winningIds;
            auction.status = AuctionStatus.Finalized;
            auction.winnerDeterminationTime = block.timestamp;
            
            // Award special smart matching bonus
            uint256 providerId = offers[bestOfferId].providerId;
            address providerAddress = registry.providerIdToAddress(providerId);
            if (providerAddress != address(0)) {
                awardProviderPoints(providerAddress, 25, "smart_match_perfect_fit");
            }
            
            address commuterAddress = registry.commuterIdToAddress(auction.commuterId);
            if (commuterAddress != address(0)) {
                awardCommuterPoints(commuterAddress, 15, "smart_match_success");
            }
            
            emit SmartMatchingCompleted(auctionId, bestOfferId);
            emit AuctionFinalized(auctionId, winningIds);
            
            return true;
        }
        
        return false;
    }
    
    // ================ View Functions ================
    
    /**
     * @dev Get offers for a specific auction
     * @param auctionId Auction ID
     * @return Array of offers
     */
    function getOffersByAuction(uint256 auctionId) public view returns (Offer[] memory) {
        return offersByAuction[auctionId];
    }
    
    /**
     * @dev Get the winning offers for an auction
     * @param auctionId Auction ID
     * @return Array of winning offers
     */
    function getWinningOffers(uint256 auctionId) public view auctionExists(auctionId) returns (Offer[] memory) {
        Auction storage auction = auctions[auctionId];
        require(auction.status == AuctionStatus.Finalized, "Auction is not finalized");
        
        uint256[] memory winningIds = auction.winningOfferIds;
        Offer[] memory winningOffers = new Offer[](winningIds.length);
        
        for (uint256 i = 0; i < winningIds.length; i++) {
            winningOffers[i] = offersByAuction[auctionId][winningIds[i]];
        }
        
        return winningOffers;
    }


    function getAuctionInfo(uint256 auctionId) public view auctionExists(auctionId) returns (
        uint256 requestId,
        uint256 commuterId,
        AuctionStatus status,
        AuctionType auctionType,
        uint256 startTime,
        uint256 endTime,
        uint256 offerCount,
        uint256 reservePrice,
        bool requiresVerified
    ) {
        Auction storage auction = auctions[auctionId];
        return (
            auction.requestId,
            auction.commuterId,
            auction.status,
            auction.auctionType,
            auction.startTime,
            auction.endTime,
            auction.offerCount,
            auction.reservePrice,
            auction.requiresVerified
        );
    }
    
    /**
     * @dev Get service confirmation details
     * @param auctionId Auction ID
     * @param offerId Offer ID
     * @return Confirmation details or empty struct if none exists
     */
    function getServiceConfirmation(uint256 auctionId, uint256 offerId) public view returns (Confirmation memory) {
        return serviceConfirmations[auctionId][offerId];
    }
    
    /**
     * @dev Get auction statistics
     * @return activeCount Number of active auctions
     * @return finishedCount Number of finished auctions
     * @return averageOffers Average number of offers per auction
     */
    function getAuctionStats() public view returns (
        uint256 activeCount,
        uint256 finishedCount,
        uint256 averageOffers
    ) {
        activeCount = 0;
        finishedCount = 0;
        uint256 totalOffers = 0;
        
        for (uint256 i = 1; i <= numberOfAuctions; i++) {
            if (auctions[i].status == AuctionStatus.Active) {
                activeCount++;
            } else if (auctions[i].status == AuctionStatus.Finalized) {
                finishedCount++;
                totalOffers += auctions[i].offerCount;
            }
        }
        
        averageOffers = finishedCount > 0 ? totalOffers / finishedCount : 0;
        
        return (activeCount, finishedCount, averageOffers);
    }
    
    // ================ Reward Functions ================
    
    /**
     * @dev Calculate reward amount based on commuter's history
     * @param commuterId Commuter ID
     * @return Reward amount
     */
    function calculateRewardAmount(uint256 commuterId) internal view returns (uint256) {
        // This could include logic based on:
        // - User's past participation
        // - Frequency of usage
        // - Profile type (e.g., subscription vs. pay-as-you-go)
        
        // For simplicity, return a fixed amount for now
        return 10;
    }
    
    /**
     * @dev Award points to a commuter
     * @param commuter Commuter address
     * @param points Points to award
     * @param reason Reason for points
     */
    function awardCommuterPoints(address commuter, uint256 points, string memory reason) internal {
        commuterBonusPoints[commuter] += points;
        emit RewardIssued(commuter, points, reason);
    }
    
    /**
     * @dev Award points to a provider
     * @param provider Provider address
     * @param points Points to award
     * @param reason Reason for points
     */
    function awardProviderPoints(address provider, uint256 points, string memory reason) internal {
        providerBonusPoints[provider] += points;
        emit RewardIssued(provider, points, reason);
    }
    
    /**
     * @dev Award bonus points (admin only)
     * @param user User address
     * @param points Points to award
     * @param reason Reason for points
     * @param isProvider Whether user is a provider
     */
    function awardBonusPoints(address user, uint256 points, string memory reason, bool isProvider) public onlyAdmin {
        if (isProvider) {
            awardProviderPoints(user, points, reason);
        } else {
            awardCommuterPoints(user, points, reason);
        }
    }
    
    /**
     * @dev Get points balance for a user
     * @param user User address
     * @param isProvider Whether user is a provider
     * @return Points balance
     */
    function getPointsBalance(address user, bool isProvider) public view returns (uint256) {
        if (isProvider) {
            return providerBonusPoints[user];
        } else {
            return commuterBonusPoints[user];
        }
    }
    
    // ================ Admin Functions ================
    
    /**
     * @dev Set platform fee percentage
     * @param newFeePercentage New fee percentage
     */
    function setPlatformFee(uint256 newFeePercentage) public onlyAdmin {
        require(newFeePercentage <= 10, "Fee cannot exceed 10%");
        platformFeePercentage = newFeePercentage;
    }
    
    /**
     * @dev Enable or disable platform fees
     * @param enabled Whether fees are enabled
     */
    function setFeesEnabled(bool enabled) public onlyAdmin {
        feesEnabled = enabled;
    }
    
    /**
     * @dev Set auction duration limits
     * @param minDuration Minimum auction duration
     * @param maxDuration Maximum auction duration
     * @param defaultDuration Default auction duration
     */
    function setAuctionDurations(
        uint256 minDuration,
        uint256 maxDuration,
        uint256 defaultDuration
    ) public onlyAdmin {
        require(minDuration <= defaultDuration, "Default must be >= minimum");
        require(defaultDuration <= maxDuration, "Default must be <= maximum");
        
        minAuctionDuration = minDuration;
        maxAuctionDuration = maxDuration;
        defaultAuctionDuration = defaultDuration;
    }
    
    /**
     * @dev Emergency pause/unpause of the platform
     * @param paused Whether to pause operations
     */
    function setEmergencyPause(bool paused) public onlyAdmin {
        emergencyPause = paused;
    }
    
    /**
     * @dev Change admin address
     * @param newAdmin New admin address
     */
    function changeAdmin(address newAdmin) public onlyAdmin {
        require(newAdmin != address(0), "Invalid address");
        admin = newAdmin;
    }
    
    /**
     * @dev Cancel auctions in bulk (emergency function)
     * @param auctionIds Array of auction IDs to cancel
     * @param reason Cancellation reason
     */
    function bulkCancelAuctions(uint256[] memory auctionIds, string memory reason) public onlyAdmin {
        for (uint256 i = 0; i < auctionIds.length; i++) {
            uint256 auctionId = auctionIds[i];
            if (auctions[auctionId].auctionId == auctionId && auctions[auctionId].status == AuctionStatus.Active) {
                auctions[auctionId].status = AuctionStatus.Cancelled;
                emit AuctionCancelled(auctionId, reason);
            }
        }
    }
}