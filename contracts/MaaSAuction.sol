// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./MaaSRegistry.sol";
import "./MaaSRequest.sol";

contract MaaSAuction {
    address public owner;
    address public marketplace; // Facade/Marketplace
    MaaSRegistry public registry;
    MaaSRequest  public request;

    struct Offer {
        uint256 providerId;
        string  contentHash;    // IPFS/URI hash for offer payload
        address submittedBy;
    }

    struct MatchResult {
        bool    exists;
        uint256 requestId;
        uint256 offerId;        // chosen offer (index in offersByRequest[requestId])
        uint256 providerId;
        uint256 priceWei;       // optional price field stored on-chain if needed
        bool    completed;      // service completion confirmed
    }

    // requestId => offers
    mapping(uint256 => Offer[]) public offersByRequest;
    // requestId => chosen match
    mapping(uint256 => MatchResult) public matchByRequest;

    event MarketplaceSet(address indexed marketplace);
    event OfferSubmitted(uint256 indexed requestId, uint256 indexed offerId, uint256 indexed providerId, string contentHash);
    event MatchRecorded(uint256 indexed requestId, uint256 indexed offerId, uint256 providerId, uint256 priceWei);
    event CompletionConfirmed(uint256 indexed requestId);

    modifier onlyOwner() { require(msg.sender == owner, "Not owner"); _; }
    modifier onlyMarketplace() { require(msg.sender == marketplace, "Not marketplace"); _; }

    constructor(address registryAddr, address requestAddr) {
        owner    = msg.sender;
        registry = MaaSRegistry(registryAddr);
        request  = MaaSRequest(requestAddr);
    }

    function setMarketplaceAddress(address mkt) external onlyOwner {
        require(mkt != address(0), "zero");
        marketplace = mkt;
        emit MarketplaceSet(marketplace);
    }

    /// @notice Marketplace posts an offer hash on behalf of a provider (or provider directly via marketplace)
    function submitOfferHash(uint256 requestId, uint256 providerId, string calldata contentHash)
        external
        onlyMarketplace
        returns (uint256 offerId)
    {
        (address providerAddr, ) = registry.getProvider(providerId);
        require(providerAddr != address(0), "provider not registered");
        require(bytes(contentHash).length > 0, "empty hash");

        offersByRequest[requestId].push(Offer({
            providerId: providerId,
            contentHash: contentHash,
            submittedBy: tx.origin
        }));
        offerId = offersByRequest[requestId].length - 1;

        emit OfferSubmitted(requestId, offerId, providerId, contentHash);
    }

    /// @notice Marketplace records the chosen match
    function recordMatchResult(uint256 requestId, uint256 offerId, uint256 providerId, uint256 priceWei)
        external
        onlyMarketplace
    {
        require(!matchByRequest[requestId].exists, "already matched");
        require(offerId < offersByRequest[requestId].length, "bad offerId");

        matchByRequest[requestId] = MatchResult({
            exists: true,
            requestId: requestId,
            offerId: offerId,
            providerId: providerId,
            priceWei: priceWei,
            completed: false
        });

        emit MatchRecorded(requestId, offerId, providerId, priceWei);
    }

    /// @notice Marketplace confirms that the service was completed (after off-chain validation)
    function confirmCompletion(uint256 requestId) external onlyMarketplace {
        require(matchByRequest[requestId].exists, "no match");
        require(!matchByRequest[requestId].completed, "already completed");
        matchByRequest[requestId].completed = true;
        emit CompletionConfirmed(requestId);
    }

    // Views
    function getOffers(uint256 requestId) external view returns (Offer[] memory) {
        return offersByRequest[requestId];
    }

    function getMatchResult(uint256 requestId) external view returns (MatchResult memory) {
        return matchByRequest[requestId];
    }
}
