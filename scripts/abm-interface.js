// abm-interface.js  (Ethers v5)
const { ethers } = require("ethers");
const fs = require("fs");
const path = require("path");

// ---------- Load deployment addresses ----------
const ADDR_PATH = path.resolve("./deployed/simplified.json");
if (!fs.existsSync(ADDR_PATH)) {
  throw new Error("Missing deployed/simplified.json. Deploy first.");
}
const ADDR = JSON.parse(fs.readFileSync(ADDR_PATH, "utf-8"));

// ---------- ABIs ----------
const FACADE_ABI = [
  "function owner() view returns (address)",
  "function marketplaceAPI() view returns (address)",
  "function setMarketplaceAPI(address api) external",
  "function registerAsCommuter(uint256 commuterId, address account) external",
  "function registerAsProvider(uint256 providerId, address account, uint8 mode) external",
  "function submitRequestHash(uint256 commuterId, string contentHash) external returns (uint256 requestId)",
  "function setRequestStatus(uint256 requestId, uint8 status) external",
  "function submitOfferHash(uint256 requestId, uint256 providerId, string contentHash) external returns (uint256 offerId)",
  "function recordMatch(uint256 requestId, uint256 offerId, uint256 providerId, uint256 priceWei) external",
  "function confirmCompletion(uint256 requestId) external"
];

const REQUEST_ABI = [
  "function getRequestHash(uint256 requestId) view returns (string)",
  "function lastRequestId() view returns (uint256)"
];

const AUCTION_ABI = [
  "function getMatchResult(uint256 requestId) view returns (tuple(bool exists,uint256 requestId,uint256 offerId,uint256 providerId,uint256 priceWei,bool completed))",
  "function getOffers(uint256 requestId) view returns (tuple(uint256 providerId,string contentHash,address submittedBy)[])"
];

// ---------- Provider & signer ----------
/**
 * @param {Object} opts
 * @param {string} opts.rpc - RPC url (e.g., http://127.0.0.1:8545)
 * @param {string} opts.privateKey - EOA used as marketplaceAPI
 */
function makeClients({ rpc, privateKey }) {
  if (!rpc || !privateKey) throw new Error("rpc and privateKey are required");

  const provider = new ethers.providers.JsonRpcProvider(rpc);
  const signer = new ethers.Wallet(privateKey, provider);

  const facade = new ethers.Contract(ADDR.Facade, FACADE_ABI, signer);
  const request = new ethers.Contract(ADDR.Request, REQUEST_ABI, provider); // read-only ok
  const auction = new ethers.Contract(ADDR.Auction, AUCTION_ABI, provider); // read-only ok

  return { provider, signer, facade, request, auction, ADDR };
}

// ---------- Identity registration (owner-only on Facade) ----------
async function registerCommuter({ facade }, commuterId, account) {
  const tx = await facade.registerAsCommuter(commuterId, account);
  const rc = await tx.wait();
  return rc.transactionHash;
}

async function registerProvider({ facade }, providerId, account, mode /* uint8 */) {
  const tx = await facade.registerAsProvider(providerId, account, mode);
  const rc = await tx.wait();
  return rc.transactionHash;
}

// ---------- Request lifecycle (API-only) ----------
async function submitRequestHash({ facade }, commuterId, contentHash) {
  const tx = await facade.submitRequestHash(commuterId, contentHash);
  await tx.wait();
  return true;
}

async function submitRequestHashAndGetId({ facade, request }, commuterId, contentHash) {
  const tx = await facade.submitRequestHash(commuterId, contentHash);
  await tx.wait();
  const id = await request.lastRequestId();
  return Number(id.toString());
}

async function setRequestStatus({ facade }, requestId, statusEnum) {
  const tx = await facade.setRequestStatus(requestId, statusEnum);
  const rc = await tx.wait();
  return rc.transactionHash;
}

// ---------- Offers & matching (API-only) ----------
async function submitOfferHash({ facade, auction }, requestId, providerId, contentHash) {
  const tx = await facade.submitOfferHash(requestId, providerId, contentHash);
  await tx.wait();

  const offers = await auction.getOffers(requestId);
  return offers.length - 1;
}

async function recordMatch({ facade }, requestId, offerId, providerId, priceWei) {
  const price = ethers.BigNumber.isBigNumber(priceWei)
    ? priceWei
    : ethers.BigNumber.from(priceWei.toString());

  const tx = await facade.recordMatch(requestId, offerId, providerId, price);
  const rc = await tx.wait();
  return rc.transactionHash;
}

async function confirmCompletion({ facade }, requestId) {
  const tx = await facade.confirmCompletion(requestId);
  const rc = await tx.wait();
  return rc.transactionHash;
}

// ---------- Read helpers ----------
async function getRequestHash({ request }, requestId) {
  return request.getRequestHash(requestId);
}

async function getMatchResult({ auction }, requestId) {
  const res = await auction.getMatchResult(requestId);
  return {
    exists: res.exists,
    requestId: Number(res.requestId.toString()),
    offerId: Number(res.offerId.toString()),
    providerId: Number(res.providerId.toString()),
    priceWei: res.priceWei.toString(),
    completed: res.completed
  };
}

module.exports = {
  makeClients,
  registerCommuter, registerProvider,
  submitRequestHash, submitRequestHashAndGetId,
  setRequestStatus, submitOfferHash,
  recordMatch, confirmCompletion,
  getRequestHash, getMatchResult
};
