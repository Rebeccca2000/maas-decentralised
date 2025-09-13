// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./MaaSRegistry.sol";

contract MaaSRequest {
    enum Status { None, Created, Matched, Completed, Cancelled }

    address public owner;
    address public marketplace;     // Facade/Marketplace authority for writes
    MaaSRegistry public registry;

    struct Request {
        uint256 commuterId;     // who created it (off-chain identity)
        string  contentHash;    // IPFS/URI hash for full payload stored off-chain
        Status  status;
        address createdBy;      // EOA that submitted via marketplace
    }

    uint256 public lastRequestId;
    mapping(uint256 => Request) public requests; // requestId => Request
    mapping(uint256 => uint256[]) public requestsByCommuter; // commuterId => requestIds

    event MarketplaceSet(address indexed marketplace);
    event RequestCreated(uint256 indexed requestId, uint256 indexed commuterId, string contentHash);
    event RequestStatusUpdated(uint256 indexed requestId, Status status);

    modifier onlyOwner() { require(msg.sender == owner, "Not owner"); _; }
    modifier onlyMarketplace() { require(msg.sender == marketplace, "Not marketplace"); _; }

    constructor(address registryAddr) {
        owner = msg.sender;
        registry = MaaSRegistry(registryAddr);
    }

    function setMarketplaceAddress(address mkt) external onlyOwner {
        require(mkt != address(0), "zero");
        marketplace = mkt;
        emit MarketplaceSet(marketplace);
    }

    /// @notice Marketplace submits a minimal request with an off-chain hash/URI
    function createRequestWithHash(uint256 commuterId, string calldata contentHash)
        external
        onlyMarketplace
        returns (uint256 requestId)
    {
        require(bytes(contentHash).length > 0, "empty hash");
        // Optional: verify commuter is registered
        require(registry.getCommuter(commuterId) != address(0), "commuter not registered");

        requestId = ++lastRequestId;

        requests[requestId] = Request({
            commuterId: commuterId,
            contentHash: contentHash,
            status: Status.Created,
            createdBy: tx.origin
        });

        requestsByCommuter[commuterId].push(requestId);
        emit RequestCreated(requestId, commuterId, contentHash);
    }

    function updateStatus(uint256 requestId, Status status) external onlyMarketplace {
        require(requestId > 0 && requestId <= lastRequestId, "bad id");
        requests[requestId].status = status;
        emit RequestStatusUpdated(requestId, status);
    }

    // View helpers
    function getRequestHash(uint256 requestId) external view returns (string memory) {
        return requests[requestId].contentHash;
    }
}
