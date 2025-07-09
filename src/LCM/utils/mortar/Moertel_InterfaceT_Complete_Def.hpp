// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "Moertel_InterfaceT.hpp"
#include "Moertel_UtilsT.hpp"

#if defined(HAVE_MOERTEL_MPI)
#include <Teuchos_DefaultMpiComm.hpp>
#else
#include <Teuchos_DefaultSerialComm.hpp>
#endif

/*----------------------------------------------------------------------*
 |  finalize construction of this interface                             |
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool
MoertelT::MOERTEL_TEMPLATE_CLASS(InterfaceT)::Complete()
{
  if (IsComplete()) {
    if (OutLevel() > 0)
      std::cout << "MoertelT: ***WRN*** MoertelT::InterfaceT::InterfaceComplete:\n"
                << "MoertelT: ***WRN*** InterfaceComplete() was called before, do "
                   "nothing\n"
                << "MoertelT: ***WRN*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    return true;
  }

  // check for NULL entries in maps
  bool ok = true;
  for (int i = 0; i < 2; ++i) {
    std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)>>::const_iterator curr;
    for (curr = node_[i].begin(); curr != node_[i].end(); ++curr) {
      if (curr->second == Teuchos::null) {
        std::cout << "***ERR*** MoertelT::InterfaceT::Complete:\n"
                  << "***ERR*** Interface # " << Id_ << ":\n"
                  << "***ERR*** found NULL entry in map of nodes\n"
                  << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
        ok = false;
      }
    }
  }
  for (int i = 0; i < 2; ++i) {
    std::map<int, Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)>>::const_iterator curr;
    for (curr = seg_[i].begin(); curr != seg_[i].end(); ++curr) {
      if (curr->second == Teuchos::null) {
        std::cout << "***ERR*** MoertelT::Interface::Complete:\n"
                  << "***ERR*** Interface # " << Id_ << ":\n"
                  << "***ERR*** found NULL entry in map of segments\n"
                  << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
        ok = false;
      }
    }
  }
  int lok = ok;
  int gok = 1;
  Teuchos::reduceAll<LO, int>(*gcomm_, Teuchos::REDUCE_MIN, 1, &lok, &gok);
  if (!gok) return false;

  // check whether all nodes for segments are present
  // (take in account that node might be on different processor)
  // this test is expensive and does not scale. It is therefore only performed
  // when user requests a high output level
#if 1
  if (OutLevel() > 9) {
    for (int proc = 0; proc < gcomm_->getSize(); ++proc) {
      for (int side = 0; side < 2; ++side) {
        // create length of list of all nodes adjacent to segments on proc
        int sendsize = 0;
        if (proc == gcomm_->getRank()) {
          std::map<int, Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)>>::const_iterator curr;
          for (curr = seg_[side].begin(); curr != seg_[side].end(); ++curr) sendsize += curr->second->Nnode();
        }
        Teuchos::broadcast<LO, int>(*gcomm_, proc, 1, &sendsize);

        // create list of all nodes adjacent to segments on proc
        std::vector<int> ids(sendsize);
        if (proc == gcomm_->getRank()) {
          std::map<int, Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)>>::const_iterator curr;
          int                                                                                     counter = 0;
          for (curr = seg_[side].begin(); curr != seg_[side].end(); ++curr) {
            const int* segids = curr->second->NodeIds();
            for (int i = 0; i < curr->second->Nnode(); ++i) ids[counter++] = segids[i];
          }
        }
        Teuchos::broadcast<LO, int>(*gcomm_, proc, sendsize, &ids[0]);

        // check on all processors for nodes in ids
        std::vector<int> foundit(sendsize);
        std::vector<int> gfoundit(sendsize);
        for (int i = 0; i < sendsize; ++i) {
          foundit[i] = 0;
          if (node_[side].find(ids[i]) != node_[side].end()) foundit[i] = 1;
        }
        Teuchos::reduceAll<LO, int>(*gcomm_, Teuchos::REDUCE_MAX, sendsize, &foundit[0], &gfoundit[0]);
        for (int i = 0; i < sendsize; ++i) {
          if (gfoundit[i] != 1) {
            if (gcomm_->getRank() == proc)
              std::cout << "***ERR*** MoertelT::Interface::Complete:\n"
                        << "***ERR*** cannot find segment's node # " << ids[i] << "\n"
                        << "***ERR*** in map of all nodes on all procs\n"
                        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
            ids.clear();
            foundit.clear();
            gfoundit.clear();
            gcomm_->barrier();
            return false;
          }
        }

        // tidy up
        ids.clear();
        foundit.clear();
        gfoundit.clear();
      }  // for (int size=0; side<2; ++side)
    }  // for (int proc=0; proc<gcomm_->NumProc(); ++proc)
  }
#endif
  // find all procs that have business on this interface (own nodes/segments)
  // build a Teuchos_comm that contains only those procs
  // this intra-communicator will be used to handle most stuff on this
  // interface so the interface will not block all other procs
  {
#if defined(HAVE_MOERTEL_MPI)
    std::vector<int> lin(gcomm_->getSize());
    std::vector<int> gin(gcomm_->getSize());
    for (int i = 0; i < gcomm_->getSize(); ++i) lin[i] = 0;

    // check ownership of any segments
    for (int i = 0; i < 2; ++i)
      if (seg_[i].size() != 0) {
        lin[gcomm_->getRank()] = 1;
        break;
      }
    // check ownership of any nodes
    for (int i = 0; i < 2; ++i)
      if (node_[i].size() != 0) {
        lin[gcomm_->getRank()] = 1;
        break;
      }
    Teuchos::reduceAll<LO, int>(*gcomm_, Teuchos::REDUCE_MAX, gcomm_->getSize(), &lin[0], &gin[0]);
    lin.clear();

    // typecast the Teuchos_Comm to Teuchos_MpiComm
    const Teuchos::MpiComm<LO>* teuchosmpicomm = dynamic_cast<const Teuchos::MpiComm<LO>*>(gcomm_.get());
    if (!teuchosmpicomm) {
      std::stringstream oss;
      oss << "***ERR*** MoertelT::Interface::Complete:\n"
          << "***ERR*** Interface " << Id() << ": Teuchos_Comm is not a Teuchos_MpiComm\n"
          << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
      throw MoertelT::ReportError(oss);
    }

    // split the communicator into participating and none-participating procs
    int color;
    int key = gcomm_->getRank();
    // I am taking part in the new comm if I have any ownership
    if (gin[gcomm_->getRank()]) color = 0;
    // I am not taking part in the new comm
    else
      color = MPI_UNDEFINED;

    // tidy up
    gin.clear();

    // create the local communicator
    lcomm_ = gcomm_->split(color, key);
#else  // the easy serial case
    const Teuchos::SerialComm<LO>* serialcomm = dynamic_cast<const Teuchos::SerialComm<LO>*>(gcomm_.get());
    if (!serialcomm) {
      std::stringstream oss;
      oss << "***ERR*** MoertelT::Interface::Complete:\n"
          << "***ERR*** Interface " << Id() << ": Teuchos::Comm is not a Teuchos::SerialComm\n"
          << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
      throw MoertelT::ReportError(oss);
    }
    lcomm_ = Teuchos::rcp(new Teuchos::SerialComm<LO>(*serialcomm));
#endif
  }

  // create a map of all nodes to there PID (process id)
  if (lcomm_ != Teuchos::null)
    for (int proc = 0; proc < lcomm_->getSize(); ++proc) {
      int lnnodes = 0;
      if (proc == lcomm_->getRank()) lnnodes = node_[0].size() + node_[1].size();
      Teuchos::broadcast<LO, int>(*lcomm_, proc, 1, &lnnodes);
      std::vector<int> ids(lnnodes);
      if (proc == lcomm_->getRank()) {
        std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)>>::const_iterator curr;
        int                                                                                  counter = 0;
        for (int side = 0; side < 2; ++side)
          for (curr = node_[side].begin(); curr != node_[side].end(); ++curr) ids[counter++] = curr->first;
      }
      Teuchos::broadcast<LO, int>(*lcomm_, proc, lnnodes, &ids[0]);
      for (int i = 0; i < lnnodes; ++i) nodePID_.insert(std::pair<int, int>(ids[i], proc));
      ids.clear();
    }

  // create a map of all segments to there PID (process id)
  if (lcomm_ != Teuchos::null)
    for (int proc = 0; proc < lcomm_->getSize(); ++proc) {
      int lnsegs = 0;
      if (proc == lcomm_->getRank()) lnsegs = seg_[0].size() + seg_[1].size();
      Teuchos::broadcast<LO, int>(*lcomm_, proc, 1, &lnsegs);
      std::vector<int> ids(lnsegs);
      if (proc == lcomm_->getRank()) {
        std::map<int, Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)>>::const_iterator curr;
        int                                                                                     counter = 0;
        for (int side = 0; side < 2; ++side)
          for (curr = seg_[side].begin(); curr != seg_[side].end(); ++curr) ids[counter++] = curr->first;
      }
      Teuchos::broadcast<LO, int>(*lcomm_, proc, lnsegs, &ids[0]);
      for (int i = 0; i < lnsegs; ++i) segPID_.insert(std::pair<int, int>(ids[i], proc));
      ids.clear();
    }

  // set isComplete_ flag
  // we set it here already as we will be using some methods that require it
  // from now on
  isComplete_ = true;

  // make the nodes know there adjacent segments
  // find max number of nodes to a segment
  if (lcomm_ != Teuchos::null) {
    int lmaxnnode = 0;
    int gmaxnnode = 0;
    for (int side = 0; side < 2; ++side) {
      std::map<int, Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)>>::const_iterator scurr;
      for (scurr = seg_[side].begin(); scurr != seg_[side].end(); ++scurr)
        if (lmaxnnode < scurr->second->Nnode()) lmaxnnode = scurr->second->Nnode();
    }
    Teuchos::reduceAll<LO, int>(*lcomm_, Teuchos::REDUCE_MAX, 1, &lmaxnnode, &gmaxnnode);

    // loop all procs and broadcast their adjacency
    for (int proc = 0; proc < lcomm_->getSize(); ++proc) {
      // local number of segments
      int lnseg = 0;
      if (proc == lcomm_->getRank()) lnseg = seg_[0].size() + seg_[1].size();
      Teuchos::broadcast<LO, int>(*lcomm_, proc, 1, &lnseg);

      // allocate vector to hold adjacency
      int              offset = gmaxnnode + 2;
      int              size   = lnseg * offset;
      std::vector<int> adj(size);

      // proc fills adjacency vector adj and broadcasts
      if (proc == lcomm_->getRank()) {
        int count = 0;
        for (int side = 0; side < 2; ++side) {
          std::map<int, Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)>>::const_iterator scurr;
          for (scurr = seg_[side].begin(); scurr != seg_[side].end(); ++scurr) {
            Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)> seg = scurr->second;
            adj[count]                                                   = seg->Id();
            adj[count + 1]                                               = seg->Nnode();
            const int* ids                                               = seg->NodeIds();
            for (int i = 0; i < seg->Nnode(); ++i) adj[count + 2 + i] = ids[i];
            count += offset;
          }
        }
      }
      Teuchos::broadcast<LO, int>(*lcomm_, proc, size, &adj[0]);

      // all procs read adj and add segment to the nodes they own
      int count = 0;
      for (int i = 0; i < lnseg; ++i) {
        int segid = adj[count];
        int nnode = adj[count + 1];
        for (int j = 0; j < nnode; ++j) {
          int nid = adj[count + 2 + j];
          if (lcomm_->getRank() == NodePID(nid)) {
            // I own this node, so set the segment segid in it
            Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> node = GetNodeViewLocal(nid);
            if (node == Teuchos::null) {
              std::stringstream oss;
              oss << "***ERR*** MoertelT::Interface::Complete:\n"
                  << "***ERR*** cannot find node " << nid << "\n"
                  << "***ERR*** in map of all nodes on this proc\n"
                  << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
              throw MoertelT::ReportError(oss);
            }
            node->AddSegment(segid);
          } else
            continue;
        }
        count += offset;
      }
      adj.clear();
    }  // for (int proc=0; proc<lcomm_->NumProc(); ++proc)
  }  // if (lComm())

  // build redundant segments and nodes
  if (lcomm_ != Teuchos::null) {
    int ok = 0;
    ok += RedundantSegments(0);
    ok += RedundantSegments(1);
    ok += RedundantNodes(0);
    ok += RedundantNodes(1);
    if (ok != 4) {
      std::stringstream oss;
      oss << "***ERR*** MoertelT::Interface::Complete:\n"
          << "***ERR*** building of redundant information failed\n"
          << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
      throw MoertelT::ReportError(oss);
    }
  }

  // make topology segments <-> nodes for each side
  if (lcomm_ != Teuchos::null) BuildNodeSegmentTopology();

  // delete distributed nodes and segments
  for (int i = 0; i < 2; ++i) {
    seg_[i].clear();
    node_[i].clear();
  }

  // we are done
  // note that there might not be any functions on the interface yet
  // they still have to be set

  return ok;
}
