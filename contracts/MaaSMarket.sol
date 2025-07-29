// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "./MaaSNFT.sol";

/**
 * @title MaaSMarket
 * @dev Secondary marketplace for trading mobility service NFTs
 * Includes advanced features like dynamic pricing, rewards, fee discounts,
 * and sophisticated search functionality
 */
contract MaaSMarket {
    // ================ Data Structures ================

    /**
     * @dev Struct for a market listing with advanced pricing features
     */
    struct SellRequest {
        uint256 tokenId;                // The NFT token ID being sold
        uint256 price;                  // Current price (may change with dynamic pricing)
        address seller;                 // Address of the seller
        bool isSold;                    // Whether the NFT has been sold
        uint256 listingTime;            // When the NFT was listed
        
        // Dynamic pricing fields
        uint256 initialPrice;           // Starting price for dynamic pricing
        uint256 finalPrice;             // Final price after decay period
        uint256 priceDecayStartTime;    // When price decay begins
        uint256 priceDecayEndTime;      // When price reaches finalPrice
        
        // Service metadata fields
        uint256 departureTime;          // When the service starts
        uint256[] originCoordinates;    // Origin location coordinates [x,y]
        uint256[] destCoordinates;      // Destination location coordinates [x,y]
        uint8 serviceType;              // Type of service (bus, train, etc.)
    }

    /**
     * @dev Structured parameters for advanced listing searches
     */
    struct SearchParams {
        uint256 minPrice;
        uint256 maxPrice;
        uint256 minDeparture;
        uint256 maxDeparture;
        uint256[] originArea;           // [x, y, radius] for origin area search
        uint256[] destArea;             // [x, y, radius] for destination area search
        address provider;               // Filter by specific provider
        uint8 serviceType;              // Filter by transport type
        bool includeExpired;            // Whether to include expired listings
    }

    // ================ State Variables ================

    MaaSNFT public nftContract;         // The NFT contract for mobility services
    IERC20 public paymentToken;         // ERC20 token used for payments
    
    SellRequest[] public market;        // All market listings
    mapping(uint256 => uint256) public tokenIdToMarketIndex; // Find a listing by NFT ID
    mapping(uint256 => bool) public isTokenListed;          // Track if NFT is listed
    
    address public admin;               // Admin account
    uint256 public marketFeePercentage = 1; // 1% base fee
    
    // Reward and discount tracking
    mapping(address => uint256) public marketActivityPoints;    // Points earned from market activity
    mapping(address => uint256) public userTransactionVolume;   // Track user's total transaction volume
    mapping(address => uint256) public userListingCount;        // Number of listings by user
    mapping(address => uint256) public userPurchaseCount;       // Number of purchases by user

    // ================ Events ================
    
    event NFTListed(uint256 indexed tokenId, uint256 initialPrice, uint256 finalPrice, address indexed seller);
    event NFTPurchased(uint256 indexed tokenId, uint256 price, address indexed buyer, address indexed seller);
    event ListingCancelled(uint256 indexed tokenId, address indexed seller);
    event MarketFeeUpdated(uint256 newFeePercentage);
    event MarketActivityReward(address indexed user, uint256 points, string activityType);
    event PriceDynamicUpdate(uint256 indexed tokenId, uint256 newPrice);

    // ================ Constructor ================
    
    /**
     * @dev Set up the marketplace with required contracts
     * @param _nftAddress Address of the MaaSNFT contract
     * @param _paymentTokenAddress Address of the ERC20 token used for payments
     */
    constructor(address _nftAddress, address _paymentTokenAddress) {
        nftContract = MaaSNFT(_nftAddress);
        paymentToken = IERC20(_paymentTokenAddress);
        admin = msg.sender;
    }

    // ================ Core Marketplace Functions ================
    
    /**
     * @dev List an NFT for sale with fixed price
     * @param tokenId The ID of the NFT to list
     * @param price The fixed selling price
     */
    function listNFTForSale(uint256 tokenId, uint256 price) public {
        // Verify sender owns the NFT
        require(nftContract.ownerOf(tokenId) == msg.sender, "Not the NFT owner");
        require(!isTokenListed[tokenId], "Token already listed");
        
        // Ensure the NFT is approved for transfer
        require(
            nftContract.isApprovedForAll(msg.sender, address(this)) || 
            nftContract.getApproved(tokenId) == address(this),
            "Market not approved to transfer NFT"
        );
        
        // Get service info from NFT
        MaaSNFT.ServiceNFT memory service = nftContract.getServiceNFT(tokenId);

        // Check if service has expired
        bool isExpired = nftContract.isExpired(service.tokenId); 
        require(!isExpired, "Cannot list expired service");
        
        // Add to market with fixed pricing (initial = final price)
        uint256 marketIndex = market.length;
        market.push(SellRequest({
            tokenId: tokenId,
            price: price,
            seller: msg.sender,
            isSold: false,
            listingTime: block.timestamp,
            initialPrice: price,
            finalPrice: price,
            priceDecayStartTime: 0,
            priceDecayEndTime: 0,
            departureTime: service.startTime,
            originCoordinates: service.originCoordinates,
            destCoordinates: service.destinationCoordinates,
            serviceType: service.serviceType
        }));
        
        // Track listing
        tokenIdToMarketIndex[tokenId] = marketIndex;
        isTokenListed[tokenId] = true;
        
        // Update seller's listing count and award points
        userListingCount[msg.sender] += 1;
        marketActivityPoints[msg.sender] += 3; // 3 points for listing
        
        emit NFTListed(tokenId, price, price, msg.sender);
        emit MarketActivityReward(msg.sender, 3, "listing");
    }
    
    /**
     * @dev List an NFT with dynamic pricing that changes over time
     * @param tokenId The ID of the NFT to list
     * @param initialPrice Starting price
     * @param finalPrice Ending price after decay period
     * @param decayDuration Time in seconds for price to decay from initial to final
     */
    function listNFTWithDynamicPricing(
        uint256 tokenId, 
        uint256 initialPrice,
        uint256 finalPrice,
        uint256 decayDuration
    ) public {
        // Verify sender owns the NFT
        require(nftContract.ownerOf(tokenId) == msg.sender, "Not the NFT owner");
        require(!isTokenListed[tokenId], "Token already listed");
        require(initialPrice >= finalPrice, "Initial price must be >= final price");
        
        // Ensure the NFT is approved for transfer
        require(
            nftContract.isApprovedForAll(msg.sender, address(this)) || 
            nftContract.getApproved(tokenId) == address(this),
            "Market not approved to transfer NFT"
        );
        
        // Get service info from NFT
        MaaSNFT.ServiceNFT memory service = nftContract.getServiceNFT(tokenId);

        // Check if service has expired
        bool isExpired = nftContract.isExpired(service.tokenId);
        require(!isExpired, "Cannot list expired service");
        
        // Add to market with dynamic pricing
        uint256 marketIndex = market.length;
        market.push(SellRequest({
            tokenId: tokenId,
            price: initialPrice, // Current price starts at initial
            seller: msg.sender,
            isSold: false,
            listingTime: block.timestamp,
            initialPrice: initialPrice,
            finalPrice: finalPrice,
            priceDecayStartTime: block.timestamp,
            priceDecayEndTime: block.timestamp + decayDuration,
            departureTime: service.startTime,
            originCoordinates: service.originCoordinates,
            destCoordinates: service.destinationCoordinates,
            serviceType: service.serviceType
        }));
        
        // Track listing
        tokenIdToMarketIndex[tokenId] = marketIndex;
        isTokenListed[tokenId] = true;
        
        // Update seller's listing count and award points
        userListingCount[msg.sender] += 1;
        marketActivityPoints[msg.sender] += 5; // 5 points for dynamic pricing (bonus)
        
        emit NFTListed(tokenId, initialPrice, finalPrice, msg.sender);
        emit MarketActivityReward(msg.sender, 5, "dynamic_listing");
    }
    
    /**
     * @dev Purchase an NFT from the marketplace
     * @param tokenId The ID of the NFT to purchase
     */
    function purchaseNFT(uint256 tokenId) public {
        require(isTokenListed[tokenId], "Token not listed for sale");
        
        uint256 marketIndex = tokenIdToMarketIndex[tokenId];
        SellRequest storage sellRequest = market[marketIndex];
        
        require(!sellRequest.isSold, "NFT already sold");
        
        // Update current price if using dynamic pricing
        uint256 currentPrice = getCurrentPrice(marketIndex);
        sellRequest.price = currentPrice; // Update the stored price
        
        // Calculate fees with volume-based discounts
        uint256 marketFee = calculateMarketFee(msg.sender, currentPrice);
        uint256 sellerAmount = currentPrice - marketFee;
        
        // Calculate royalties for original provider
        (address royaltyReceiver, uint256 royaltyAmount) = nftContract.royaltyInfo(tokenId, currentPrice);
        if (royaltyAmount > 0 && royaltyReceiver != address(0)) {
            // Adjust seller amount to account for royalties
            sellerAmount = sellerAmount - royaltyAmount;
            
            // Transfer royalties to original provider
            require(
                paymentToken.transferFrom(msg.sender, royaltyReceiver, royaltyAmount),
                "Royalty payment failed"
            );
        }
        
        // Transfer payment to seller
        require(
            paymentToken.transferFrom(msg.sender, sellRequest.seller, sellerAmount),
            "Payment to seller failed"
        );
        
        // Transfer market fee
        if (marketFee > 0) {
            require(
                paymentToken.transferFrom(msg.sender, admin, marketFee),
                "Payment of market fee failed"
            );
        }
        
        // Transfer NFT
        nftContract.safeTransferFrom(sellRequest.seller, msg.sender, tokenId);
        
        // Update listing status
        sellRequest.isSold = true;
        isTokenListed[tokenId] = false;
        
        // Update transaction history and rewards
        userTransactionVolume[msg.sender] += currentPrice;
        userPurchaseCount[msg.sender] += 1;
        
        // Award points to both buyer and seller
        marketActivityPoints[msg.sender] += 5; // Buyer points
        marketActivityPoints[sellRequest.seller] += 2; // Seller points
        
        emit NFTPurchased(tokenId, currentPrice, msg.sender, sellRequest.seller);
        emit MarketActivityReward(msg.sender, 5, "purchase");
        emit MarketActivityReward(sellRequest.seller, 2, "sale");
    }
    
    /**
     * @dev Cancel a listing
     * @param tokenId The ID of the NFT to cancel listing
     */
    function cancelListing(uint256 tokenId) public {
        require(isTokenListed[tokenId], "Token not listed for sale");
        
        uint256 marketIndex = tokenIdToMarketIndex[tokenId];
        SellRequest storage sellRequest = market[marketIndex];
        
        require(sellRequest.seller == msg.sender || msg.sender == admin, 
                "Only the seller or admin can cancel");
        
        // Update listing
        sellRequest.isSold = true;
        isTokenListed[tokenId] = false;
        
        emit ListingCancelled(tokenId, sellRequest.seller);
    }
    
    /**
     * @dev Update market fee percentage (admin only)
     * @param newFeePercentage New fee percentage (0-10)
     */
    function updateMarketFee(uint256 newFeePercentage) public {
        require(msg.sender == admin, "Only admin can update fee");
        require(newFeePercentage <= 10, "Fee too high");
        
        marketFeePercentage = newFeePercentage;
        
        emit MarketFeeUpdated(newFeePercentage);
    }

    // ================ Dynamic Pricing Functions ================
    
    /**
     * @dev Calculate current price based on time-based dynamic pricing
     * @param marketIndex Index of the listing in the market array
     * @return Current price based on time decay
     */
    function getCurrentPrice(uint256 marketIndex) public view returns (uint256) {
        SellRequest storage listing = market[marketIndex];
        
        // If not using dynamic pricing or decay period hasn't started
        if (listing.initialPrice == listing.finalPrice || 
            block.timestamp < listing.priceDecayStartTime) {
            return listing.initialPrice;
        }
        
        // If past the decay end time
        if (block.timestamp > listing.priceDecayEndTime) {
            return listing.finalPrice;
        }
        
        // Calculate current price based on linear decay
        uint256 decayPeriod = listing.priceDecayEndTime - listing.priceDecayStartTime;
        uint256 timeElapsed = block.timestamp - listing.priceDecayStartTime;
        uint256 priceDifference = listing.initialPrice - listing.finalPrice;
        
        return listing.initialPrice - (priceDifference * timeElapsed / decayPeriod);
    }
    
    /**
     * @dev Manually update prices for all dynamic listings
     * Useful to refresh prices before searches or displays
     */
    function refreshDynamicPrices() public {
        for (uint256 i = 0; i < market.length; i++) {
            SellRequest storage listing = market[i];
            
            // Skip sold listings or fixed price listings
            if (listing.isSold || listing.initialPrice == listing.finalPrice) {
                continue;
            }
            
            uint256 newPrice = getCurrentPrice(i);
            if (newPrice != listing.price) {
                listing.price = newPrice;
                emit PriceDynamicUpdate(listing.tokenId, newPrice);
            }
        }
    }

    // ================ Fee Calculation Functions ================
    
    /**
     * @dev Calculate market fee with volume-based discounts
     * @param buyer Address of the buyer
     * @param price Purchase price
     * @return Fee amount with applicable discounts
     */
    function calculateMarketFee(address buyer, uint256 price) internal view returns (uint256) {
        uint256 volume = userTransactionVolume[buyer];
        uint256 purchaseCount = userPurchaseCount[buyer];
        
        uint256 discountPercentage = 0;
        
        // Volume-based discounts
        if (volume > 10 ether) {
            discountPercentage += 30; // 30% discount
        } else if (volume > 5 ether) {
            discountPercentage += 15; // 15% discount
        }
        
        // Frequency-based discounts
        if (purchaseCount > 10) {
            discountPercentage += 10; // Additional 10% discount
        } else if (purchaseCount > 5) {
            discountPercentage += 5; // Additional 5% discount
        }
        
        // Cap total discount at 50%
        discountPercentage = discountPercentage > 50 ? 50 : discountPercentage;
        
        // Calculate discounted fee
        uint256 baseAmount = price * marketFeePercentage / 100;
        uint256 discountAmount = baseAmount * discountPercentage / 100;
        
        return baseAmount - discountAmount;
    }

    // ================ Search Functions ================
    
    /**
     * @dev Get all active listings
     * @return Array of active (unsold) listings
     */
    function getActiveListings() public view returns (SellRequest[] memory) {
        // Count active listings
        uint256 activeCount = 0;
        for (uint256 i = 0; i < market.length; i++) {
            if (!market[i].isSold) {
                activeCount++;
            }
        }
        
        // Create result array
        SellRequest[] memory activeListings = new SellRequest[](activeCount);
        
        // Fill result array
        uint256 index = 0;
        for (uint256 i = 0; i < market.length; i++) {
            if (!market[i].isSold) {
                activeListings[index] = market[i];
                index++;
            }
        }
        
        return activeListings;
    }
    
    /**
     * @dev Basic search with price and departure time filtering
     * @param minPrice Minimum price (0 for no minimum)
     * @param maxPrice Maximum price (0 for no maximum)
     * @param minDeparture Minimum departure time (0 for no minimum)
     * @param maxDeparture Maximum departure time (0 for no maximum)
     * @return Array of matching listings
     */
    function searchListings(
        uint256 minPrice, 
        uint256 maxPrice,
        uint256 minDeparture,
        uint256 maxDeparture
    ) public view returns (SellRequest[] memory) {
        // Update all dynamic prices before searching
        SellRequest[] memory active = getActiveListings();
        
        // Count matching listings
        uint256 matchCount = 0;
        for (uint256 i = 0; i < active.length; i++) {
            // Get current price if dynamic
            uint256 currentPrice = active[i].price;
            if (active[i].initialPrice != active[i].finalPrice) {
                currentPrice = getCurrentPrice(tokenIdToMarketIndex[active[i].tokenId]);
            }
            
            bool priceMatch = (minPrice == 0 || currentPrice >= minPrice) && 
                             (maxPrice == 0 || currentPrice <= maxPrice);
                             
            bool departureMatch = (minDeparture == 0 || active[i].departureTime >= minDeparture) &&
                                 (maxDeparture == 0 || active[i].departureTime <= maxDeparture);
            
            if (priceMatch && departureMatch) {
                matchCount++;
            }
        }
        
        // Create result array
        SellRequest[] memory matches = new SellRequest[](matchCount);
        
        // Fill result array
        uint256 index = 0;
        for (uint256 i = 0; i < active.length; i++) {
            // Get current price if dynamic
            uint256 currentPrice = active[i].price;
            if (active[i].initialPrice != active[i].finalPrice) {
                currentPrice = getCurrentPrice(tokenIdToMarketIndex[active[i].tokenId]);
            }
            
            bool priceMatch = (minPrice == 0 || currentPrice >= minPrice) && 
                             (maxPrice == 0 || currentPrice <= maxPrice);
                             
            bool departureMatch = (minDeparture == 0 || active[i].departureTime >= minDeparture) &&
                                 (maxDeparture == 0 || active[i].departureTime <= maxDeparture);
            
            if (priceMatch && departureMatch) {
                matches[index] = active[i];
                index++;
            }
        }
        
        return matches;
    }
    
    /**
     * @dev Advanced search with multiple filtering parameters
     * @param params Structured search parameters
     * @return Array of matching listings
     */
    function advancedSearch(SearchParams memory params) public view returns (SellRequest[] memory) {
        SellRequest[] memory active = getActiveListings();
        
        // Count matching listings
        uint256 matchCount = 0;
        for (uint256 i = 0; i < active.length; i++) {
            // Check if expired service should be excluded
            if (!params.includeExpired) {
                bool isExpired = nftContract.isExpired(active[i].tokenId);
                if (isExpired) continue;
            }
            
            // Get current price if dynamic
            uint256 currentPrice = active[i].price;
            if (active[i].initialPrice != active[i].finalPrice) {
                currentPrice = getCurrentPrice(tokenIdToMarketIndex[active[i].tokenId]);
            }
            
            // Check price range
            bool priceMatch = (params.minPrice == 0 || currentPrice >= params.minPrice) && 
                             (params.maxPrice == 0 || currentPrice <= params.maxPrice);
            
            // Check departure time
            bool departureMatch = (params.minDeparture == 0 || active[i].departureTime >= params.minDeparture) &&
                                 (params.maxDeparture == 0 || active[i].departureTime <= params.maxDeparture);
            
            // Check service type
            bool typeMatch = params.serviceType == 0 || active[i].serviceType == params.serviceType;
            
            // Check provider if specified
            bool providerMatch = true;
            if (params.provider != address(0)) {
                MaaSNFT.ServiceNFT memory service = nftContract.getServiceNFT(active[i].tokenId);
                address providerAddress = nftContract.getProviderAddress(service.providerId);
                providerMatch = (providerAddress == params.provider);
            }
            
            // Check origin area if specified
            bool originMatch = true;
            if (params.originArea.length == 3) {
                uint256 originX = active[i].originCoordinates[0];
                uint256 originY = active[i].originCoordinates[1];
                uint256 searchX = params.originArea[0];
                uint256 searchY = params.originArea[1];
                uint256 radius = params.originArea[2];
                
                // Calculate squared distance
                uint256 dx = originX > searchX ? originX - searchX : searchX - originX;
                uint256 dy = originY > searchY ? originY - searchY : searchY - originY;
                uint256 distanceSquared = dx*dx + dy*dy;
                
                originMatch = distanceSquared <= radius*radius;
            }
            
            // Check destination area if specified
            bool destMatch = true;
            if (params.destArea.length == 3) {
                uint256 destX = active[i].destCoordinates[0];
                uint256 destY = active[i].destCoordinates[1];
                uint256 searchX = params.destArea[0];
                uint256 searchY = params.destArea[1];
                uint256 radius = params.destArea[2];
                
                // Calculate squared distance
                uint256 dx = destX > searchX ? destX - searchX : searchX - destX;
                uint256 dy = destY > searchY ? destY - searchY : searchY - destY;
                uint256 distanceSquared = dx*dx + dy*dy;
                
                destMatch = distanceSquared <= radius*radius;
            }
            
            // If all criteria match, count this listing
            if (priceMatch && departureMatch && typeMatch && providerMatch && originMatch && destMatch) {
                matchCount++;
            }
        }
        
        // Create result array
        SellRequest[] memory matches = new SellRequest[](matchCount);
        
        // Fill result array
        uint256 index = 0;
        for (uint256 i = 0; i < active.length; i++) {
            // Check if expired service should be excluded
            if (!params.includeExpired) {
                bool isExpired = nftContract.isExpired(active[i].tokenId);
                if (isExpired) continue;
            }
            
            // Get current price if dynamic
            uint256 currentPrice = active[i].price;
            if (active[i].initialPrice != active[i].finalPrice) {
                currentPrice = getCurrentPrice(tokenIdToMarketIndex[active[i].tokenId]);
            }
            
            // Check price range
            bool priceMatch = (params.minPrice == 0 || currentPrice >= params.minPrice) && 
                             (params.maxPrice == 0 || currentPrice <= params.maxPrice);
            
            // Check departure time
            bool departureMatch = (params.minDeparture == 0 || active[i].departureTime >= params.minDeparture) &&
                                 (params.maxDeparture == 0 || active[i].departureTime <= params.maxDeparture);
            
            // Check service type
            bool typeMatch = params.serviceType == 0 || active[i].serviceType == params.serviceType;
            
            // Check provider if specified
            bool providerMatch = true;
            if (params.provider != address(0)) {
                MaaSNFT.ServiceNFT memory service = nftContract.getServiceNFT(active[i].tokenId);
                address providerAddress = nftContract.getProviderAddress(service.providerId);
                providerMatch = (providerAddress == params.provider);
            }
            
            // Check origin area if specified
            bool originMatch = true;
            if (params.originArea.length == 3) {
                uint256 originX = active[i].originCoordinates[0];
                uint256 originY = active[i].originCoordinates[1];
                uint256 searchX = params.originArea[0];
                uint256 searchY = params.originArea[1];
                uint256 radius = params.originArea[2];
                
                // Calculate squared distance
                uint256 dx = originX > searchX ? originX - searchX : searchX - originX;
                uint256 dy = originY > searchY ? originY - searchY : searchY - originY;
                uint256 distanceSquared = dx*dx + dy*dy;
                
                originMatch = distanceSquared <= radius*radius;
            }
            
            // Check destination area if specified
            bool destMatch = true;
            if (params.destArea.length == 3) {
                uint256 destX = active[i].destCoordinates[0];
                uint256 destY = active[i].destCoordinates[1];
                uint256 searchX = params.destArea[0];
                uint256 searchY = params.destArea[1];
                uint256 radius = params.destArea[2];
                
                // Calculate squared distance
                uint256 dx = destX > searchX ? destX - searchX : searchX - destX;
                uint256 dy = destY > searchY ? destY - searchY : searchY - destY;
                uint256 distanceSquared = dx*dx + dy*dy;
                
                destMatch = distanceSquared <= radius*radius;
            }
            
            // If all criteria match, add to results
            if (priceMatch && departureMatch && typeMatch && providerMatch && originMatch && destMatch) {
                matches[index] = active[i];
                index++;
            }
        }
        
        return matches;
    }

    // ================ Reward System Functions ================
    
    /**
     * @dev Get points balance for a user
     * @param user Address to check
     * @return Current points balance
     */
    function getPointsBalance(address user) public view returns (uint256) {
        return marketActivityPoints[user];
    }
    
    /**
     * @dev Redeem points for benefits (future implementation)
     * @param amount Amount of points to redeem
     * @param benefitType String identifier for the benefit type
     */
    function redeemPoints(uint256 amount, string memory benefitType) public {
        require(marketActivityPoints[msg.sender] >= amount, "Insufficient points");
        
        // Reduce points balance
        marketActivityPoints[msg.sender] -= amount;
        
        // Emit event for tracking
        emit MarketActivityReward(msg.sender, amount, string(abi.encodePacked("redeem_", benefitType)));
        
        // Future: Could implement different benefit types here
    }
    
    /**
     * @dev Award bonus points to a user (admin only)
     * @param user Address to receive points
     * @param amount Points to award
     * @param reason Reason for bonus
     */
    function awardBonusPoints(address user, uint256 amount, string memory reason) public {
        require(msg.sender == admin, "Only admin can award bonus points");
        
        marketActivityPoints[user] += amount;
        
        emit MarketActivityReward(user, amount, string(abi.encodePacked("bonus_", reason)));
    }
}