// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./MaaSRegistry.sol";
import "./MaaSRequest.sol";
import "./MaaSAuction.sol";

/// @title MaaSFacade (Simplified)
/// @notice Very thin facade the Marketplace/Backend uses to call into Registry/Request/Auction
contract MaaSFacade {
    address public owner;
    address public marketplaceAPI; // optional signer used by your backend

    MaaSRegistry public registry;
    MaaSRequest  public request;
    MaaSAuction  public auction;

    event MarketplaceAPISet(address indexed api);

    modifier onlyOwner() { require(msg.sender == owner, "Not owner"); _; }
    modifier onlyAPI()   { require(msg.sender == marketplaceAPI, "Not API"); _; }

    constructor(address registryAddr, address requestAddr, address auctionAddr) {
        owner    = msg.sender;
        registry = MaaSRegistry(registryAddr);
        request  = MaaSRequest(requestAddr);
        auction  = MaaSAuction(auctionAddr);
    }

    function setMarketplaceAPI(address api) external onlyOwner {
        require(api != address(0), "zero");
        marketplaceAPI = api;
        emit MarketplaceAPISet(api);
    }


    // --- Registration (owner-only to keep on-chain surface tiny/controlled) ---
    function registerAsCommuter(uint256 commuterId, address account) external onlyOwner {
        registry.registerCommuter(commuterId, account);
    }

    function registerAsProvider(uint256 providerId, address account, uint8 mode) external onlyOwner {
        registry.registerProvider(providerId, account, mode);
    }

    // --- Request lifecycle (API-only) ---
    function submitRequestHash(uint256 commuterId, string calldata contentHash)
        external
        onlyAPI
        returns (uint256 requestId)
    {
        requestId = request.createRequestWithHash(commuterId, contentHash);
    }

    function setRequestStatus(uint256 requestId, MaaSRequest.Status status) external onlyAPI {
        request.updateStatus(requestId, status);
    }

    // --- Offers & matching (API-only) ---
    function submitOfferHash(uint256 requestId, uint256 providerId, string calldata contentHash)
        external
        onlyAPI
        returns (uint256 offerId)
    {
        offerId = auction.submitOfferHash(requestId, providerId, contentHash);
    }

    function recordMatch(uint256 requestId, uint256 offerId, uint256 providerId, uint256 priceWei)
        external
        onlyAPI
    {
        auction.recordMatchResult(requestId, offerId, providerId, priceWei);
        // Optionally set request status to Matched in one call:
        request.updateStatus(requestId, MaaSRequest.Status.Matched);
    }

    function confirmCompletion(uint256 requestId) external onlyAPI {
        auction.confirmCompletion(requestId);
        // Optionally set request status to Completed:
        request.updateStatus(requestId, MaaSRequest.Status.Completed);
    }
}
