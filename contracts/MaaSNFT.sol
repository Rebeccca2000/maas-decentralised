// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "./MaaSRegistry.sol";
import "./MaaSRequest.sol";
import "./MaaSAuction.sol";

/**
 * @title MaaSNFT
 * @dev NFT contract for tokenizing mobility services
 * Includes enhanced features for verification, expiration, bundles,
 * and provider royalties
 */
contract MaaSNFT is ERC721URIStorage {
    // ================ Data Structures ================
    
    /**
     * @dev Main struct for service NFT details
     */
    struct ServiceNFT {
        uint256 tokenId;              // NFT token ID
        uint256 requestId;            // Related request ID
        uint256 providerId;           // Service provider ID
        string routeDetails;          // JSON string with route details
        uint256 price;                // Original purchase price
        uint256 startTime;            // Service start time
        uint256 duration;             // Service duration in seconds
        
        // Enhanced fields
        address originalProvider;     // Original service provider address
        uint256 royaltyPercentage;    // Royalty percentage for secondary sales
        bool verified;                // Whether service has been verified
        uint256 qualityScore;         // Rating/quality score (0-100)
        string[] serviceTags;         // Tags for categorization
        uint256 validUntil;           // Timestamp when service expires
        bool expired;                 // Whether service has expired
        uint8 serviceType;            // Type of transport (matches enum in Registry)
        uint256[] originCoordinates;  // Origin coordinates [x,y]
        uint256[] destinationCoordinates; // Destination coordinates [x,y]
    }
    
    /**
     * @dev Bundle of multiple services
     */
    struct ServiceBundle {
        uint256 bundleId;             // Unique bundle ID
        uint256[] componentTokenIds;  // Component NFT token IDs
        string name;                  // Bundle name
        uint256 price;                // Bundle price (with discount)
        address owner;                // Bundle owner
        bool active;                  // Whether bundle is active
    }
    
    // ================ State Variables ================
    
    mapping(uint256 => ServiceNFT) public serviceNFTs;      // NFT ID to service details
    uint256 private _tokenIdCounter = 0;                      // Counter for NFT IDs
    
    mapping(uint256 => ServiceBundle) public bundles;        // Bundle ID to bundle details
    uint256 private _bundleCounter = 0;                       // Counter for bundle IDs
    
    mapping(address => bool) public verifiers;               // Authorized service verifiers
    mapping(address => uint256) public serviceCompletionPoints; // Reward points
    
    // Contract references
    IERC20 public paymentToken;
    MaaSRegistry public registry;
    MaaSRequest public requestContract;
    MaaSAuction public auctionContract;
    address public admin;
    
    // ================ Events ================
    
    event ServiceTokenized(uint256 requestId, uint256 tokenId, string tokenURI);
    event ServicePaid(uint256 requestId, uint256 providerId, uint256 amount);
    event ServiceVerified(uint256 tokenId, uint256 score, address verifier);
    event ServiceExpired(uint256 tokenId);
    event ServiceRated(uint256 tokenId, uint256 score, address rater);
    event BundleCreated(uint256 bundleId, address creator, uint256[] tokenIds);
    event BundlePurchased(uint256 bundleId, address buyer, uint256 price);
    event RewardPointsIssued(address user, uint256 points, string activity);
    
    // ================ Constructor ================
    
    /**
     * @dev Initialize the NFT contract with required dependencies
     * @param _registryAddress Address of registry contract
     * @param _requestAddress Address of request contract
     * @param _auctionAddress Address of auction contract
     * @param _paymentTokenAddress Address of payment token contract
     */
    constructor(
        address _registryAddress, 
        address _requestAddress, 
        address _auctionAddress,
        address _paymentTokenAddress
    ) ERC721("MaaS Service NFT", "MAAS") {
        registry = MaaSRegistry(_registryAddress);
        requestContract = MaaSRequest(_requestAddress);
        auctionContract = MaaSAuction(_auctionAddress);
        paymentToken = IERC20(_paymentTokenAddress);
        admin = msg.sender;
        
        // Add admin as a verifier
        verifiers[admin] = true;
    }
    // ================ Core NFT Functions ================
    
    /**
     * @dev Pay for a mobility service
     * @param requestId ID of the travel request
     * @param providerId ID of the service provider
     * @param amount Payment amount
     * @return Success status
     */
    function payForService(uint256 requestId, uint256 providerId, uint256 amount) public returns (bool) {
        // Get request basic info
        (uint256 _requestId, uint256 commuterId, , , , , , ) = 
            requestContract.getRequestBasicInfo(requestId);
        
        // Verify request exists and belongs to sender
        require(_requestId == requestId, "Request does not exist");
        
        (uint256 senderCommuterId,,,,,,,,,) = registry.getCommuter(msg.sender); 
        require(senderCommuterId == commuterId, "Request doesn't belong to sender");
        
        // Get provider address
        address providerAddress = registry.providerIdToAddress(providerId);
        require(providerAddress != address(0), "Provider not found");
        
        // Transfer payment to provider
        require(
            paymentToken.transferFrom(msg.sender, providerAddress, amount),
            "Payment failed"
        );
        
        // Update request status
        requestContract.updateRequestStatus(requestId, MaaSRequest.Status.ServiceSelected);
        
        // Award points to commuter for using service
        serviceCompletionPoints[msg.sender] += 10;
        emit RewardPointsIssued(msg.sender, 10, "service_payment");
        
        emit ServicePaid(requestId, providerId, amount);
        return true;
    }
    
    /**
     * @dev Create a service NFT for a completed transaction
     * @param requestId ID of the travel request
     * @param providerId ID of the service provider
     * @param routeDetails JSON string with route details
     * @param price Service price
     * @param startTime Service start time
     * @param duration Service duration
     * @param tokenURI URI for token metadata
     * @return NFT token ID
     */
    function mintServiceNFT(
        uint256 requestId, 
        uint256 providerId,
        string memory routeDetails,
        uint256 price,
        uint256 startTime,
        uint256 duration,
        string memory tokenURI
    ) public returns (uint256) {
        // Get request basic info
        (uint256 _requestId, uint256 commuterId, uint256[] memory origin, uint256[] memory destination, , , , uint8 modeType) = 
            requestContract.getRequestBasicInfo(requestId);
        
        // Verify request exists and belongs to sender or is a provider
        require(_requestId == requestId, "Request does not exist");
        
        (uint256 testSenderCommuterId,,,,,,,,,) = registry.getCommuter(msg.sender);
        MaaSRegistry.Provider memory provider = registry.getProvider(msg.sender);
        
        require(testSenderCommuterId == commuterId || provider.providerId == providerId, 
                "Not authorized to mint NFT for this service");
        
        // Get provider address for royalties
        address providerAddress = registry.providerIdToAddress(providerId);
        
        // Mint NFT
        _tokenIdCounter++;
        uint256 newItemId = _tokenIdCounter;
        _mint(msg.sender, newItemId);
        _setTokenURI(newItemId, tokenURI);
        
        // Store NFT details
        string[] memory emptyTags = new string[](0);
        
        serviceNFTs[newItemId] = ServiceNFT({
            tokenId: newItemId,
            requestId: requestId,
            providerId: providerId,
            routeDetails: routeDetails,
            price: price,
            startTime: startTime,
            duration: duration,
            originalProvider: providerAddress,
            royaltyPercentage: 5, // 5% royalty by default
            verified: false,
            qualityScore: 0,
            serviceTags: emptyTags,
            validUntil: startTime + duration, // Service expires after duration
            expired: false,
            serviceType: modeType,
            originCoordinates: origin,
            destinationCoordinates: destination
        });
        
        // Award points for creating an NFT
        serviceCompletionPoints[msg.sender] += 5;
        emit RewardPointsIssued(msg.sender, 5, "mint_nft");
        
        emit ServiceTokenized(requestId, newItemId, tokenURI);
        
        return newItemId;
    }
    
    /**
     * @dev Get detailed information about a service NFT
     * @param tokenId NFT token ID
     * @return Service NFT details
     */
    function getServiceNFT(uint256 tokenId) public view returns (ServiceNFT memory) {
        require(_exists(tokenId), "Token does not exist");
        return serviceNFTs[tokenId];
    }
    
    /**
     * @dev Get provider address for a service NFT
     * @param providerId Provider ID
     * @return Provider address
     */
    function getProviderAddress(uint256 providerId) public view returns (address) {
        return registry.providerIdToAddress(providerId);
    }

    // ================ Service Verification Functions ================
    
    /**
     * @dev Add a new authorized verifier
     * @param verifier Address to authorize as verifier
     */
    function addVerifier(address verifier) public {
        require(msg.sender == admin, "Only admin can add verifiers");
        verifiers[verifier] = true;
    }
    
    /**
     * @dev Remove a verifier
     * @param verifier Address to remove as verifier
     */
    function removeVerifier(address verifier) public {
        require(msg.sender == admin, "Only admin can remove verifiers");
        verifiers[verifier] = false;
    }
    
    /**
     * @dev Verify a service and assign quality score
     * @param tokenId NFT token ID
     * @param score Quality score (0-100)
     */
    function verifyService(uint256 tokenId, uint256 score) public {
        require(verifiers[msg.sender], "Not an authorized verifier");
        require(score <= 100, "Score must be 0-100");
        require(_exists(tokenId), "Token does not exist");
        
        ServiceNFT storage nft = serviceNFTs[tokenId];
        nft.verified = true;
        nft.qualityScore = score;
        
        emit ServiceVerified(tokenId, score, msg.sender);
    }
    
    /**
     * @dev Rate a service as a user
     * @param tokenId NFT token ID
     * @param score User rating (0-100)
     */
    function rateService(uint256 tokenId, uint256 score) public {
        require(score <= 100, "Score must be 0-100");
        require(_exists(tokenId), "Token does not exist");
        
        // User must have owned this NFT at some point
        require(
            ownerOf(tokenId) == msg.sender || 
            balanceOf(msg.sender) > 0, // User has some NFTs (simplification)
            "Only users can rate services"
        );
        
        ServiceNFT storage nft = serviceNFTs[tokenId];
        
        // Update quality score (weighted average with existing score)
        if (nft.qualityScore > 0) {
            nft.qualityScore = (nft.qualityScore + score) / 2;
        } else {
            nft.qualityScore = score;
        }
        
        // Award points for rating
        serviceCompletionPoints[msg.sender] += 2;
        emit RewardPointsIssued(msg.sender, 2, "rate_service");
        
        emit ServiceRated(tokenId, score, msg.sender);
    }
    
    /**
     * @dev Add service tags for better categorization
     * @param tokenId NFT token ID
     * @param tags Array of tag strings
     */
    function addServiceTags(uint256 tokenId, string[] memory tags) public {
        require(_exists(tokenId), "Token does not exist");
        require(
            ownerOf(tokenId) == msg.sender || 
            msg.sender == admin ||
            verifiers[msg.sender],
            "Not authorized to add tags"
        );
        
        serviceNFTs[tokenId].serviceTags = tags;
    }

    // ================ Service Expiration Functions ================
    
    /**
     * @dev Check if a service has expired and update status
     * @param tokenId NFT token ID
     * @return Expiration status
     */
    function checkExpiration(uint256 tokenId) public returns (bool) {
        require(_exists(tokenId), "Token does not exist");
        
        ServiceNFT storage nft = serviceNFTs[tokenId];
        if (!nft.expired && block.timestamp > nft.validUntil) {
            nft.expired = true;
            emit ServiceExpired(tokenId);
            return true;
        }
        return nft.expired;
    }
    
    /**
     * @dev Check expiration without updating state (view function)
     * @param tokenId NFT token ID
     * @return Expiration status
     */
    function isExpired(uint256 tokenId) public view returns (bool) {
        require(_exists(tokenId), "Token does not exist");
        
        ServiceNFT memory nft = serviceNFTs[tokenId];
        if (!nft.expired && block.timestamp > nft.validUntil) {
            return true;
        }
        return nft.expired;
    }
    
    /**
     * @dev Extend the validity of a service (e.g. delayed service)
     * @param tokenId NFT token ID
     * @param extensionTime Additional time in seconds
     */
    function extendValidity(uint256 tokenId, uint256 extensionTime) public {
        require(_exists(tokenId), "Token does not exist");
        require(
            msg.sender == admin ||
            registry.providerIdToAddress(serviceNFTs[tokenId].providerId) == msg.sender,
            "Only provider or admin can extend validity"
        );
        
        ServiceNFT storage nft = serviceNFTs[tokenId];
        nft.validUntil += extensionTime;
        
        // If service was expired, un-expire it
        if (nft.expired && block.timestamp <= nft.validUntil) {
            nft.expired = false;
        }
    }
    
    /**
     * @dev Before token transfer hook to prevent transferring expired services
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 tokenId,
        uint256 batchSize // Add this parameter
    ) internal override {
        super._beforeTokenTransfer(from, to, tokenId, batchSize);
        
        // Skip during minting
        if (from != address(0)) {
            bool isServiceExpired = isExpired(tokenId);
            require(!isServiceExpired, "Cannot transfer expired service");
        }
    }
    // ================ Royalty Functions ================
    
    /**
     * @dev Get royalty information for a token (EIP-2981 compatible)
     * @param tokenId NFT token ID
     * @param salePrice Sale price
     * @return receiver Royalty receiver address
     * @return royaltyAmount Royalty amount
     */
    function royaltyInfo(
        uint256 tokenId,
        uint256 salePrice
    ) external view returns (address receiver, uint256 royaltyAmount) {
        require(_exists(tokenId), "Token does not exist");
        
        ServiceNFT memory nft = serviceNFTs[tokenId];
        return (nft.originalProvider, salePrice * nft.royaltyPercentage / 100);
    }
    
    /**
     * @dev Update royalty percentage for a token
     * @param tokenId NFT token ID
     * @param percentage New royalty percentage
     */
    function updateRoyaltyPercentage(uint256 tokenId, uint256 percentage) public {
        require(_exists(tokenId), "Token does not exist");
        require(percentage <= 25, "Royalty can't exceed 25%");
        require(
            msg.sender == admin ||
            registry.providerIdToAddress(serviceNFTs[tokenId].providerId) == msg.sender,
            "Only provider or admin can update royalty"
        );
        
        serviceNFTs[tokenId].royaltyPercentage = percentage;
    }

    // ================ Service Bundle Functions ================
    
    /**
     * @dev Create a bundle of multiple services
     * @param tokenIds Array of NFT token IDs to bundle
     * @param name Bundle name
     * @param price Bundle price (with discount)
     * @return Bundle ID
     */
    function createServiceBundle(
        uint256[] memory tokenIds,
        string memory name,
        uint256 price
    ) public returns (uint256) {
        require(tokenIds.length > 1, "Bundle needs at least 2 services");
        
        // Verify ownership of all component tokens
        for (uint256 i = 0; i < tokenIds.length; i++) {
            require(ownerOf(tokenIds[i]) == msg.sender, "Must own all services");
            require(!isExpired(tokenIds[i]), "Cannot bundle expired services");
        }
        
        _bundleCounter++;
        bundles[_bundleCounter] = ServiceBundle({
            bundleId: _bundleCounter,
            componentTokenIds: tokenIds,
            name: name,
            price: price,
            owner: msg.sender,
            active: true
        });
        
        // Award points for creating a bundle
        serviceCompletionPoints[msg.sender] += 10;
        emit RewardPointsIssued(msg.sender, 10, "create_bundle");
        
        emit BundleCreated(_bundleCounter, msg.sender, tokenIds);
        
        return _bundleCounter;
    }
    
    /**
     * @dev Purchase a service bundle
     * @param bundleId Bundle ID
     */
    function purchaseBundle(uint256 bundleId) public {
        require(bundles[bundleId].bundleId > 0, "Bundle does not exist");
        require(bundles[bundleId].active, "Bundle is not active");
        require(bundles[bundleId].owner != msg.sender, "Cannot buy own bundle");
        
        ServiceBundle memory bundle = bundles[bundleId];
        
        // Verify all services still valid and owned by seller
        for (uint256 i = 0; i < bundle.componentTokenIds.length; i++) {
            uint256 tokenId = bundle.componentTokenIds[i];
            require(!isExpired(tokenId), "Bundle contains expired service");
            require(ownerOf(tokenId) == bundle.owner, "Seller no longer owns all services");
        }
        
        // Transfer payment
        require(
            paymentToken.transferFrom(msg.sender, bundle.owner, bundle.price),
            "Bundle payment failed"
        );
        
        // Transfer all NFTs
        for (uint256 i = 0; i < bundle.componentTokenIds.length; i++) {
            _transfer(bundle.owner, msg.sender, bundle.componentTokenIds[i]);
        }
        
        // Deactivate bundle
        bundles[bundleId].active = false;
        
        // Award points for using bundle feature
        serviceCompletionPoints[msg.sender] += 15;
        serviceCompletionPoints[bundle.owner] += 5;
        
        emit RewardPointsIssued(msg.sender, 15, "purchase_bundle");
        emit RewardPointsIssued(bundle.owner, 5, "sell_bundle");
        
        emit BundlePurchased(bundleId, msg.sender, bundle.price);
    }
    
    /**
     * @dev Deactivate a bundle
     * @param bundleId Bundle ID
     */
    function deactivateBundle(uint256 bundleId) public {
        require(bundles[bundleId].bundleId > 0, "Bundle does not exist");
        require(bundles[bundleId].owner == msg.sender || msg.sender == admin, 
                "Not authorized to deactivate");
        
        bundles[bundleId].active = false;
    }
    
    /**
     * @dev Get all active bundles
     * @return Array of active bundles
     */
    function getActiveBundles() public view returns (ServiceBundle[] memory) {
        // Count active bundles
        uint256 activeCount = 0;
        for (uint256 i = 1; i <= _bundleCounter; i++) {
            if (bundles[i].active) {
                activeCount++;
            }
        }
        
        // Create result array
        ServiceBundle[] memory activeBundles = new ServiceBundle[](activeCount);
        
        // Fill result array
        uint256 index = 0;
        for (uint256 i = 1; i <= _bundleCounter; i++) {
            if (bundles[i].active) {
                activeBundles[index] = bundles[i];
                index++;
            }
        }
        
        return activeBundles;
    }

    // ================ Reward System Functions ================
    
    /**
     * @dev Get points balance for a user
     * @param user Address to check
     * @return Points balance
     */
    function getRewardPoints(address user) public view returns (uint256) {
        return serviceCompletionPoints[user];
    }
    
    /**
     * @dev Award bonus points to a user
     * @param user Address to receive points
     * @param amount Points amount
     */
    function awardBonusPoints(address user, uint256 amount) public {
        require(msg.sender == admin, "Only admin can award bonus points");
        
        serviceCompletionPoints[user] += amount;
        
        emit RewardPointsIssued(user, amount, "admin_bonus");
    }
}